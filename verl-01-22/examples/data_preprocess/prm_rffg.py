# -*- coding: utf-8 -*-
"""
Convert ShareGPT-style RFF-G dialogues to AgentLoop multi-turn episode parquet.

Old behavior (Interaction-driven)
--------------------------------
- One row per conversation (ONLY a turn0 entry point)
- Pack ALL backward hints into extra_info.interaction_kwargs.backward_hints
- Rollout/Interaction drives subsequent turns

New behavior (AgentLoop-driven: step_verify_agent)
-------------------------------------------------
- One row per conversation (turn0 entry point)
- Put ALL backward hints into extra_info.subproblems (List[str])
- (Optional) Put the next gpt message after each hint into extra_info.expected_answers (List[str])
- Set agent_name = "step_verify_agent"

Then `StepVerifyAgentLoop` will consume `extra_info.subproblems` as multi-turn user inputs and perform
step-wise verification with token-level credit assignment.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from datasets import Dataset

# 1) turn0 注入的 system 指令（多轮过程中后续 user/backward 由 Interaction 继续提供）
SYSTEM_TEMPLATE = (
    "You alternate between backward (<BACKWARD>) and forward reasoning.\n"
    "For each forward response:\n"
    "  1) Add one or a few concise reasoning steps that progress the current subgoal.\n"
    "  2) When the subgoal is fully satisfied, move on to the next hint.\n"
    "  3) On the final hint, write a complete proof of the main goal.\n"
    "Do not include extra chit-chat. No JSON unless explicitly asked."
)

# Final synthesis prompt appended to subproblems to force a clean final answer.
FINAL_ANSWER_PROMPT_TEMPLATE = (
    "<BACKWARD>\n"
    "Now produce the complete final solution based on ALL previous steps.\n"
    "Requirements:\n"
    "- Write a coherent, correct end-to-end solution.\n"
    "- End with the final numeric answer wrapped EXACTLY as:\n"
    "  <ANSWER>...</ANSWER>\n"
    "- Do NOT include multiple answers.\n"
    "</BACKWARD>\n"
    "\n"
    "Problem (repeat):\n"
    "{problem}\n"
)

# 2) 读取 json/jsonl
def iter_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith("["):
            data = json.load(f)
            for obj in data:
                yield obj
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

# # 3) ground truth 解析（仅做元信息埋点，不计算reward）
# import re

# def extract_answer(text: str):
#     """
#     从字符串中提取 <ANSWER></ANSWER> 标签包裹的内容。
#     若有多个匹配，返回列表；若没有匹配，返回空列表。
#     """
#     pattern = re.compile(r"<ANSWER>(.*?)</ANSWER>", re.DOTALL | re.IGNORECASE)
#     matches = pattern.findall(text)
#     # 去掉首尾空格
#     return [m.strip() for m in matches]
import re

# 预编译：ANSWER 标签（大小写不敏感、允许换行、允许标签内外空格）
ANSWER_BLOCK_RE = re.compile(
    r"<\s*ANSWER\s*>(.*?)<\s*/\s*ANSWER\s*>",
    re.IGNORECASE | re.DOTALL
)

# 预编译：LaTeX \boxed{...}，允许空格、跨行；尽量取“最后一个盒子”为最终答案
# BOXED_STRICT_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}", re.DOTALL)
# BOXED_RELAX_RE  = re.compile(r"\\boxed\s*\{(.*?)\}", re.DOTALL)  # 兜底
BOXED_STRICT_RE = re.compile(r"\\boxed\s*\{(.*?)\}", re.DOTALL)
BOXED_RELAX_RE = re.compile(r"\\boxed\s*\{(.*?)\}", re.DOTALL)

def _strip_dollar(s: str) -> str:
    """去掉首尾成对的 $ 或 $$ 包裹（常见于 LaTeX 行内/行间公式）。"""
    t = s.strip()
    # 去掉外层 $$...$$ 或 $...$
    if t.startswith("$$") and t.endswith("$$") and len(t) >= 4:
        return t[2:-2].strip()
    if t.startswith("$") and t.endswith("$") and len(t) >= 2:
        return t[1:-1].strip()
    return t

def _extract_last_boxed(text: str) -> str | None:
    """从文本中提取最后一个 \\boxed{...} 的内容（去掉外层 $/$$ 与末尾标点）。"""
    # 先去掉外层 $ 包裹，避免影响正则匹配
    t = _strip_dollar(text)
    m_all = BOXED_STRICT_RE.findall(t)
    if not m_all:
        m_all = BOXED_RELAX_RE.findall(t)
    if not m_all:
        return None
    inner = m_all[-1].strip()
    inner = _strip_dollar(inner)
    # 去掉末尾常见符号
    inner = inner.rstrip(" .;，。；!")
    return inner

def extract_answer(text: str) -> List[str]:
    """
    提取 <ANSWER>...</ANSWER> 中的内容。
    - 若内部包含 \\boxed{...}，返回盒内内容（去掉外层 $/$$），如 '0'。
    - 否则返回去空白的原文。
    - 多个 ANSWER 标签时返回对应的列表（按出现顺序）。
    """
    if not isinstance(text, str) or not text:
        return []
    blocks = ANSWER_BLOCK_RE.findall(text)
    results: List[str] = []
    for blk in blocks:
        blk_stripped = blk.strip()
        boxed_inner = _extract_last_boxed(blk_stripped)
        if boxed_inner is not None:
            results.append(boxed_inner)
        else:
            # 无 boxed 时，尝试剥去外层 $ 再返回
            results.append(_strip_dollar(blk_stripped))
    return results



def resolve_ground_truth(conv: List[Dict[str, Any]]) -> List[str]:
    # for k in ("final_answer", "answer"):
    #     if k in conv_obj and conv_obj[k]:
    #         return str(conv_obj[k]).strip()
    # for parent in ("meta", "_raw", "extra_info"):
    #     d = conv_obj.get(parent)
    #     if isinstance(d, dict):
    #         for k in ("final_answer", "answer"):
    #             v = d.get(k)
    #             if v:
    #                 return str(v).strip()
    # rm = conv_obj.get("reward_model")
    # if isinstance(rm, dict) and rm.get("ground_truth"):
    #     return str(rm["ground_truth"]).strip()
    answers = []
    for line in conv:
        v = line.get('value', '')
        if '<ANSWER>' in v:
            ans = extract_answer(v)
            if ans:
                # return ans[0]
                answers.extend(ans)
    # import pdb;pdb.set_trace()
    if len(answers) > 0:
        return answers[-1]


    return None

# 4) 主转换：按轮展开（human backward -> gpt forward）
def build_episode_from_sharegpt(conv_obj: Dict[str, Any], data_source: str) -> List[Dict[str, Any]]:
    """
    生成 AgentLoop multi-turn episode（entry-only）数据：
    - 只输出一条 row（turn0 entry point）
    - 把所有 backward(human) 文本按顺序打包进 extra_info.subproblems
    - 把每个 hint 后面紧跟的 gpt 输出（若存在）打包进 extra_info.expected_answers
    """
    rows: List[Dict[str, Any]] = []
    conversations = conv_obj.get("conversations", [])
    if not isinstance(conversations, list) or not conversations:
        return rows

    # 提取所有 human/backward 内容（按出现顺序），以及其后第一个 gpt 回复作为参考答案（可为空）
    backward_hints: List[str] = []
    expected_answers: List[str] = []
    for i, m in enumerate(conversations):
        if not (isinstance(m, dict) and m.get("from") == "human"):
            continue
        v = m.get("value", "")
        if isinstance(v, str):
            v = v.strip()
        if not v:
            continue
        backward_hints.append(v)

        # Find the next gpt response after this human message
        ans = ""
        for j in range(i + 1, len(conversations)):
            mj = conversations[j]
            if isinstance(mj, dict) and mj.get("from") == "gpt":
                av = mj.get("value", "")
                if isinstance(av, str):
                    av = av.strip()
                ans = av or ""
                break
        expected_answers.append(ans)

    if not backward_hints:
        return rows

    gt = resolve_ground_truth(conversations)  # 仅埋点

    # 可选保留题干提示（通常第一条 human 就带 Problem）
    question_hint = backward_hints[0]

    # Append a final synthesis step to force final answer with <ANSWER> tag.
    # This makes it easier to apply final reward based on answer extraction.
    final_prompt = FINAL_ANSWER_PROMPT_TEMPLATE.format(problem=question_hint)
    backward_hints.append(final_prompt)
    # Keep alignment: expected_answers length should match subproblems length.
    expected_answers.append("")

    # 尝试保留 id（如果有）
    problem_id = conv_obj.get("problem_id") or conv_obj.get("id") or conv_obj.get("uuid")

    # turn0 prompt = system + 第一个 backward hint（通常包含 Problem + first hint）
    prompt = [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": backward_hints[0]},
    ]

    row = {
        "data_source": data_source,
        "agent_name": "step_verify_agent",
        "prompt": prompt,
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": gt,  # 仅埋点
        },
        "extra_info": {
            "turn_index": 0,
            "problem_id": problem_id,
            "question_hint": question_hint,
            # Multi-turn user inputs for StepVerifyAgentLoop
            "subproblems": backward_hints,
            # Optional reference assistant answers aligned by index (may contain empty strings)
            "expected_answers": expected_answers,
        },
    }

    rows.append(row)
    return rows

def main():
    # in_file = '/local/scratch/zqin30/projects/LLaMA-Factory/data/prm800k_rffg_1k_backforthconv3.jsonl'
    in_file = "/local/scratch/zqin30/projects/repo/verl/examples/data_preprocess/math_rffg_train_backforthconv_solutiongt.jsonl"
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default=in_file, required=False, help="Input ShareGPT JSON/JSONL")
    ap.add_argument("--out_parquet", default="math_rffg_1223/math_train_solutiongt.parquet", required=False, help="Output parquet file")
    ap.add_argument("--data_source", default="rffg-prm800k", help="Tag for data_source column")
    args = ap.parse_args()

    all_rows: List[Dict[str, Any]] = []
    conv_cnt = 0
    row_cnt  = 0

    for idx, conv in enumerate(iter_json_or_jsonl(args.in_path)):
        rows = build_episode_from_sharegpt(conv, data_source=args.data_source)
        if rows:
            # inject stable index for AgentLoop trajectory tracking
            for r in rows:
                if "extra_info" not in r or r["extra_info"] is None:
                    r["extra_info"] = {}
                r["extra_info"]["index"] = idx
            all_rows.extend(rows)
            row_cnt += len(rows)
        conv_cnt += 1

    if not all_rows:
        raise RuntimeError("No rows produced. Check ShareGPT input format.")

    ds = Dataset.from_list(all_rows)
    # import pdb;pdb.set_trace()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_parquet)), exist_ok=True)
    ds.to_parquet(args.out_parquet)
    print(f"[OK] conversations={conv_cnt}, rows={row_cnt}, saved -> {args.out_parquet}")

if __name__ == "__main__":
    main()
