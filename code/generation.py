import torch

def generate_with_logits(model, tokenizer, prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs.sequences[0][input_length:].tolist()
    scores = outputs.scores

    assert len(generated_ids) == len(scores), (
        f"Length mismatch: {len(generated_ids)} tokens vs {len(scores)} score tensors. "
        f"input_length={input_length}, total_length={outputs.sequences.shape[1]}"
    )

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return answer, scores, generated_ids
