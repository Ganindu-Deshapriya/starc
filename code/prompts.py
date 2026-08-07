def format_prompt(question, model_name, tokenizer=None):
    model_lower = model_name.lower()

    if 'phi-2' in model_lower and 'instruct' not in model_lower:
        return f"Instruct: {question}\nOutput:"

    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return question
