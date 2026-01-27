import torch


def test_step_token_credit_assignment_mean_spread():
    # response_mask: 1 for assistant tokens, 0 for tool/user/pad
    response_mask = torch.tensor([[1, 1, 1, 0, 1, 1, 0]], dtype=torch.long)
    rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)

    # Two steps:
    # - step1 covers tokens [0, 3) => 3 assistant tokens => each gets 1/3
    # - step2 covers tokens [4, 6) => 2 assistant tokens => each gets 0.5
    turn_scores = [1.0, 1.0]
    spans = [(0, 3), (4, 6)]

    resp_len = response_mask.size(1)
    for s, (start, end) in zip(turn_scores, spans, strict=False):
        start = max(0, min(resp_len, int(start)))
        end = max(0, min(resp_len, int(end)))
        per = float(s) / float(end - start)
        rm_scores[0, start:end] += per

    rm_scores = rm_scores * response_mask.to(torch.float32)

    expected = torch.tensor([[1 / 3, 1 / 3, 1 / 3, 0.0, 0.5, 0.5, 0.0]], dtype=torch.float32)
    assert torch.allclose(rm_scores, expected, atol=1e-6), (rm_scores, expected)


def test_step_plus_final_reward_additive():
    # assistant tokens at positions: 0,1,2,4,5  (total 5)
    response_mask = torch.tensor([[1, 1, 1, 0, 1, 1, 0]], dtype=torch.long)

    # step rewards same as previous test
    turn_scores = [1.0, 1.0]
    spans = [(0, 3), (4, 6)]

    rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
    resp_len = response_mask.size(1)
    for s, (start, end) in zip(turn_scores, spans, strict=False):
        start = max(0, min(resp_len, int(start)))
        end = max(0, min(resp_len, int(end)))
        rm_scores[0, start:end] += float(s) / float(end - start)

    rm_scores = rm_scores * response_mask.to(torch.float32)

    # final reward = 1.0, uniformly across ALL assistant tokens (5 tokens) => +0.2 each
    final_reward = 1.0
    denom = response_mask.to(torch.float32).sum().item()
    rm_scores = rm_scores + (final_reward / denom) * response_mask.to(torch.float32)

    expected = torch.tensor(
        [[1 / 3 + 0.2, 1 / 3 + 0.2, 1 / 3 + 0.2, 0.0, 0.5 + 0.2, 0.5 + 0.2, 0.0]],
        dtype=torch.float32,
    )
    assert torch.allclose(rm_scores, expected, atol=1e-6), (rm_scores, expected)


