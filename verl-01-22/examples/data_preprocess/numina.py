# -*- coding: utf-8 -*-
"""
Convert RFF-G JSON/JSONL (with goals + dialogue) to per-turn parquet for veRL GRPO.

Assumed input format (one JSON object per line):

{
  "problem_id": "algebra_4013",
  "source": "NuminaMath-LEAN",
  "problem": {
    "statement_nl": "...",
    "statement_latex": "...",
    "answer": "1" | "proof" | "unknown" | ...
  },
  "goals": [
    {
      "id": "ASSUMPTION",
      "kind": "assumption" | "subgoal" | "final_goal",
      "nl": "...",
      "latex": "...",
      "lean_prop": "..." | null
    },
    ...
  ],
  "edges": [...],
  "dialogue": [
    {
      "turn_id": 0,
      "role": "user" | "assistant",
      "speaker": "student" | "backward_agent" | "forward_agent",
      "kind": "problem" | "backward_step" | "forward_step",
      "from_goal_id": "G1" | null,
      "to_goal_id": "G2" | "FINAL" | null,
      "content": "..."
    },
    ...
  ]
}

Output parquet schema (each row = one forward_agent turn):

- data_source: str
- prompt: List[{"role": "system"|"user", "content": str}]
- response: str
- ability: str (e.g. "math")
- reward_model: { "style": "rule", "ground_truth": str | None }
- extra_info: {
    "turn_index": int,
    "problem_id": str,
    "problem_text": str,
    "current_from_goal_id": str | None,
    "current_to_goal_id": str | None,
    "current_goal_nl": str | None,
    "current_goal_lean": str | None,
    "is_final_step": bool,
    "backward_hint": str | None,
    "next_backward_hint": str | None,
    "expects_ready_tag": bool,
    "expects_answer_tag": bool
  }
- interaction_kwargs: {
    "name": str,
    "problem_id": str,
    "problem_text": str,
    "ground_truth": str | None,
    "final_goal_id": str | None,
    "goals": { goal_id: { "kind","nl","latex","lean_prop" } },
    "goal_order": List[{"from": str | None, "to": str | None}],
    "max_turns": int,
    "stop_regex": str,
    # 可供自定义 interaction 使用的额外信息
}

You can then plug this parquet into veRL GRPO as offline interaction data.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset

# 1) 每轮都注入的 system 指令（你可以按需再改）
SYSTEM_TEMPLATE = (
    "You are a forward reasoning agent working together with a backward planner.\n"
    "Each turn you receive a backward hint describing the current subgoal.\n"
    "For each forward response:\n"
    "  1) Add one or a few concise reasoning steps that truly progress the current subgoal.\n"
    "  2) When the current subgoal is fully satisfied or the final theorem is proved, "
    "     END your reply with <READY>.\n"
    "  3) If you can produce the final numeric or short answer, include it as <Answer>...</Answer>.\n"
    "Avoid chit-chat. Do not output JSON unless explicitly asked."
)


def iter_json_or_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """兼容读取 JSON 数组 or JSONL（一行一条 JSON）。"""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.lstrip().startswith("["):
            data = json.load(f)
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj


def build_goals_index(goals_list: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """
    把 goals 列表整理成 id -> goal_info 的 dict，并尝试找出 final_goal_id。
    """
    goals: Dict[str, Dict[str, Any]] = {}
    final_goal_id: Optional[str] = None

    for g in goals_list:
        gid = g.get("id")
        if not gid:
            continue
        goals[gid] = {
            "kind": g.get("kind"),
            "nl": (g.get("nl") or "").strip() or None,
            "latex": (g.get("latex") or "").strip() or None,
            "lean_prop": g.get("lean_prop"),
        }
        if g.get("kind") == "final_goal":
            final_goal_id = gid

    # 兜底：如果没标 final_goal，就尝试找叫 "FINAL" 的
    if final_goal_id is None and "FINAL" in goals:
        final_goal_id = "FINAL"

    return goals, final_goal_id


def resolve_ground_truth_from_problem(problem: Dict[str, Any]) -> Optional[str]:
    """
    从 problem 字段中解析 ground truth（用于最终答案奖励）。
    优先 problem['answer']，若为 'proof' / 'unknown' 等则返回 None。
    """
    ans = problem.get("answer")
    if not ans:
        return None
    ans_str = str(ans).strip()
    if ans_str.lower() in {"proof", "unknown", "none", "na"}:
        return None
    return ans_str


def collect_dialogue_turns(dialogue: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    将对话按 turn_id 排序，并切分出 backward / forward 步骤。
    """
    if not isinstance(dialogue, list):
        return [], []

    sorted_turns = sorted(dialogue, key=lambda x: x.get("turn_id", 0))

    backward_steps: List[Dict[str, Any]] = []
    forward_steps: List[Dict[str, Any]] = []

    for t in sorted_turns:
        spk = t.get("speaker")
        kind = t.get("kind")
        if spk == "backward_agent" and kind == "backward_step":
            backward_steps.append(t)
        elif spk == "forward_agent" and kind == "forward_step":
            forward_steps.append(t)

    return backward_steps, forward_steps


def build_backward_hint_maps(
    backward_steps: List[Dict[str, Any]]
) -> Tuple[Dict[Tuple[Optional[str], Optional[str]], str], Dict[str, str]]:
    """
    建立两类 backward hint 索引：
    - by_pair[(from_goal_id, to_goal_id)] -> content
      （方便精确配对某个 forward: (G_from -> G_to)）
    - by_to[to_goal_id] -> content
      （兜底：根据子目标 id 拿一条 backward 描述）
    """
    by_pair: Dict[Tuple[Optional[str], Optional[str]], str] = {}
    by_to: Dict[str, str] = {}

    for b in backward_steps:
        from_id = b.get("from_goal_id")
        to_id = b.get("to_goal_id")
        content = (b.get("content") or "").strip()
        if not content:
            continue
        by_pair[(from_id, to_id)] = content
        if to_id:
            # 多次出现时，以最后一次为准（通常链是线性的）
            by_to[to_id] = content

    return by_pair, by_to


def find_neighbor_backward_hints(
    sorted_turns: List[Dict[str, Any]],
    forward_turn_index: int,
) -> Tuple[Optional[str], Optional[str]]:
    """
    在原始排序对话中，基于位置寻找：
    - prev_backward_hint: 当前 forward 前面最近的 backward_agent.content
    - next_backward_hint: 当前 forward 后面最近的 backward_agent.content
    用于：
      - prev_backward_hint: 构造当前轮 user 提示
      - next_backward_hint: 作为 interaction_kwargs 或 extra_info 提供给下一轮
    """
    prev_hint: Optional[str] = None
    next_hint: Optional[str] = None

    # 向前找最近的 backward_agent
    for j in range(forward_turn_index - 1, -1, -1):
        t = sorted_turns[j]
        if t.get("speaker") == "backward_agent" and t.get("kind") == "backward_step":
            prev_hint = (t.get("content") or "").strip() or None
            break

    # 向后找最近的 backward_agent
    for j in range(forward_turn_index + 1, len(sorted_turns)):
        t = sorted_turns[j]
        if t.get("speaker") == "backward_agent" and t.get("kind") == "backward_step":
            next_hint = (t.get("content") or "").strip() or None
            break

    return prev_hint, next_hint


def build_rows_from_rff(conv_obj: Dict[str, Any], data_source: str) -> List[Dict[str, Any]]:
    """
    核心转换逻辑：
    - 输入：一条 RFF-G JSON（含 problem/goals/dialogue）
    - 输出：多行，每行对应一次 forward_agent turn
    """
    rows: List[Dict[str, Any]] = []

    problem_id = str(conv_obj.get("problem_id") or conv_obj.get("id") or "")
    if not problem_id:
        # 没有 id 的样本直接丢弃
        return rows

    problem = conv_obj.get("problem") or {}
    problem_text = (
        (problem.get("statement_nl") or "").strip()
        or (problem.get("statement_latex") or "").strip()
    )
    if not problem_text:
        # 没题目文本，跳过
        return rows

    goals_list = conv_obj.get("goals") or []
    goals_dict, final_goal_id = build_goals_index(goals_list)
    ground_truth_answer = resolve_ground_truth_from_problem(problem)

    dialogue = conv_obj.get("dialogue") or []
    sorted_turns = sorted(dialogue, key=lambda x: x.get("turn_id", 0))

    backward_steps, forward_steps = collect_dialogue_turns(dialogue)
    if not forward_steps:
        return rows

    # backward 索引：精确 pair + 兜底按子目标 id
    backward_by_pair, backward_by_to = build_backward_hint_maps(backward_steps)

    # 构建 goal_order：按 forward 出现顺序记录 (from,to)
    goal_order: List[Dict[str, Optional[str]]] = []
    for fs in sorted(forward_steps, key=lambda x: x.get("turn_id", 0)):
        goal_order.append(
            {
                "from": fs.get("from_goal_id"),
                "to": fs.get("to_goal_id"),
            }
        )

    # 统一的 interaction_kwargs（每一行复用）
    # 这里 name 可以在 main 里通过 data_source 派生，也可以写死
    interaction_kwargs_common = {
        "name": "rffg-math-numina",
        "problem_id": problem_id,
        "problem_text": problem_text,
        "ground_truth": ground_truth_answer,
        "final_goal_id": final_goal_id,
        "goals": goals_dict,
        "goal_order": goal_order,
        # 多轮对话的理论 max_turns：可以选 len(forward_steps)*2 或 20 等
        "max_turns": max(10, len(forward_steps) * 2),
        # 在线交互时可用：一旦生成 <READY> 或 <Answer> 就认为本轮可以结束
        "stop_regex": "<READY>|<Answer>",
    }

    # 主循环：按 turn_id 顺序遍历所有 turn，找到 forward_turn，并构造一行
    turn_index = 0
    for idx, t in enumerate(sorted_turns):
        if not (t.get("speaker") == "forward_agent" and t.get("kind") == "forward_step"):
            continue

        from_id = t.get("from_goal_id")
        to_id = t.get("to_goal_id")

        # 1) 当前轮的 backward hint（优先精确 pair 匹配，其次按子目标兜底，再次用 neighbor）
        backward_hint = None

        # 尝试使用配对的 backward_step: backward: (from=to_id, to=from_id)
        pair_key = (to_id, from_id)
        if pair_key in backward_by_pair:
            backward_hint = backward_by_pair[pair_key]

        # 若没有精确配对，则用 to_id 兜底（很多设计中 backward 的 to_goal 是“子目标”）
        if not backward_hint and to_id and to_id in backward_by_to:
            backward_hint = backward_by_to[to_id]

        # 再兜底：从对话里按位置找最近的 backward（前面那一个）
        prev_hint, next_hint_neighbor = find_neighbor_backward_hints(sorted_turns, idx)
        if not backward_hint:
            backward_hint = prev_hint

        if not backward_hint:
            # 最后兜底：直接拿当前目标的自然语言描述
            # 这里优先 from_id，表示“当前想要推进的目标”
            goal_nl = None
            if from_id and from_id in goals_dict and goals_dict[from_id]["nl"]:
                goal_nl = goals_dict[from_id]["nl"]
            elif to_id and to_id in goals_dict and goals_dict[to_id]["nl"]:
                goal_nl = goals_dict[to_id]["nl"]
            if goal_nl:
                backward_hint = f"To make progress, focus on the following goal:\n{goal_nl}"
            else:
                backward_hint = "Describe the next reasoning step toward the current goal."

        # 2) 下一轮可能用到的 backward 提示（如果 reward_fn / interaction 需要）
        next_backward_hint = next_hint_neighbor

        # 3) 构建 prompt：system + user(backward)
        if turn_index == 0:
            # 第一轮：把题目也喂进去
            user_text = f"Problem:\n{problem_text}\n\nBackward hint:\n{backward_hint}"
        else:
            user_text = f"Backward hint:\n{backward_hint}"

        prompt = [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user", "content": user_text},
        ]

        # 4) 当前 forward_agent 的示范输出（可以当作 SFT 参考）
        response_text = (t.get("content") or "").strip()

        # 5) 当前轮目标信息（给 reward_fn / 环境用）
        # 这里把“当前要推进的目标”理解为 from_goal（子目标本身），
        # 但同时也附上 to_goal 方便你自己在 reward 中做 bridge 检查。
        from_goal = goals_dict.get(from_id) if from_id else None
        to_goal = goals_dict.get(to_id) if to_id else None

        current_goal_nl = None
        current_goal_lean = None

        # 优先 regard from_id 为“当前子目标”，如果它存在自然语言/lean_prop
        if from_goal and (from_goal.get("nl") or from_goal.get("lean_prop")):
            current_goal_nl = from_goal.get("nl")
            current_goal_lean = from_goal.get("lean_prop")
        elif to_goal:
            current_goal_nl = to_goal.get("nl")
            current_goal_lean = to_goal.get("lean_prop")

        is_final_step = (to_id is not None and to_id == final_goal_id)

        extra_info = {
            "turn_index": turn_index,
            "problem_id": problem_id,
            "problem_text": problem_text,
            "current_from_goal_id": from_id,
            "current_to_goal_id": to_id,
            "current_goal_nl": current_goal_nl,
            "current_goal_lean": current_goal_lean,
            "is_final_step": bool(is_final_step),
            "backward_hint": backward_hint,
            "next_backward_hint": next_backward_hint,
            # 一般所有 step 都希望模型用 <READY> 结束本 subgoal 的 forward 部分
            "expects_ready_tag": True,
            # 但只有最后一步需要 <Answer> 标签（如果问题有答案）
            "expects_answer_tag": bool(is_final_step and ground_truth_answer is not None),
        }

        reward_model = {
            "style": "rule",
            # 这里仅埋点 final answer，具体 reward_fn 决定是否使用
            "ground_truth": ground_truth_answer,
        }

        # interaction_kwargs：对同一 problem 固定（这里每行都写一遍，veRL 会照抄）
        interaction_kwargs = dict(interaction_kwargs_common)
        # 你也可以在这里按 turn_index 加一点分支逻辑，比如首轮才启用某些辅助信息等

        extra_info["interaction_kwargs"] = interaction_kwargs  # ← 把 interaction_kwargs 放进 extra_info

        row = {
            "data_source": data_source,
            "prompt": prompt,
            "response": response_text,
            "ability": "math",
            "reward_model": reward_model,
            "extra_info": extra_info,  # ← 现在 extra_info 包含 interaction_kwargs
        }

        rows.append(row)
        turn_index += 1

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_path",
        required=True,
        help="Input RFF-G JSON/JSONL file (one sample per line or a JSON list).",
    )
    ap.add_argument(
        "--out_parquet",
        required=True,
        help="Output parquet file path.",
    )
    ap.add_argument(
        "--data_source",
        default="rffg-math-numina-data",
        help="Tag for data_source column.",
    )
    args = ap.parse_args()

    all_rows: List[Dict[str, Any]] = []
    conv_cnt = 0
    row_cnt = 0

    for conv_obj in iter_json_or_jsonl(args.in_path):
        conv_cnt += 1
        if conv_cnt < 500:
            continue
        rows = build_rows_from_rff(conv_obj, data_source=args.data_source)
        if not rows:
            continue
        all_rows.extend(rows)
        row_cnt += len(rows)

    if not all_rows:
        raise RuntimeError(
            f"No rows produced from {args.in_path}. "
            "Check that the input is RFF-G JSON with problem/goals/dialogue."
        )

    ds = Dataset.from_list(all_rows)
    import pdb;pdb.set_trace()
    out_path = os.path.abspath(args.out_parquet)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds.to_parquet(out_path)

    print(f"[OK] conversations={conv_cnt}, rows={row_cnt}, saved -> {out_path}")


if __name__ == "__main__":
    main()
