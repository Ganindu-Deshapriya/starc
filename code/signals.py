import torch
import torch.nn.functional as F

def compute_token_entropies(scores):
    entropies = []
    for logits in scores:
        logits = logits.squeeze(0)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        entropies.append(entropy)
    return entropies

def aggregate_entropy(entropies):
    if not entropies:
        return 0.0, 0.0, 0.0
    mean_H = sum(entropies) / len(entropies)
    max_H = max(entropies)
    sorted_H = sorted(entropies, reverse=True)
    top_n = max(1, len(sorted_H) // 10)
    top10_H = sum(sorted_H[:top_n]) / top_n
    return mean_H, max_H, top10_H

def compute_perplexity(scores, generated_token_ids):
    if len(scores) != len(generated_token_ids):
        min_len = min(len(scores), len(generated_token_ids))
        scores = scores[:min_len]
        generated_token_ids = generated_token_ids[:min_len]
    assert len(scores) == len(generated_token_ids)

    log_probs = []
    for logits, token_id in zip(scores, generated_token_ids):
        logits = logits.squeeze(0)
        probs = F.softmax(logits, dim=-1)
        p_actual = probs[token_id].item()
        log_probs.append(torch.log(torch.tensor(p_actual + 1e-9)).item())

    avg_neg_log = -sum(log_probs) / len(log_probs)
    ppl = torch.exp(torch.tensor(avg_neg_log)).item()
    return min(ppl, 100.0)

def compute_probability_gaps(scores):
    gaps = []
    for logits in scores:
        logits = logits.squeeze(0)
        probs = F.softmax(logits, dim=-1)
        top2_values = torch.topk(probs, 2).values
        gap = (top2_values[0] - top2_values[1]).item()
        gaps.append(gap)
    mean_gap = sum(gaps) / len(gaps)
    min_gap = min(gaps)
    return mean_gap, min_gap

def compute_delta_entropy(entropies, spike_threshold=None):
    if len(entropies) < 2:
        return 0, 0.0, [], []
    deltas = [abs(entropies[i] - entropies[i - 1]) for i in range(1, len(entropies))]
    max_delta = max(deltas)
    if spike_threshold is None:
        return 0, max_delta, [], deltas
    spike_positions = [i + 1 for i, d in enumerate(deltas) if d > spike_threshold]
    return len(spike_positions), max_delta, spike_positions, deltas
