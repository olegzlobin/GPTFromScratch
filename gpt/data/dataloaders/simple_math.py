from datasets import load_dataset


def get_math_dataset(tokenizer, n_samples=None, max_length=128):
    dataset_name = "ozlobin/simple_algebra_1m"
    if n_samples is not None:
        dataset = load_dataset(dataset_name, split=f'train[:{n_samples}]')
    else:
        dataset = load_dataset(dataset_name)

    def tokenize_function(examples):
        texts = []
        for inst, ans in zip(examples['instruction'], examples['answer']):
            text = f"{inst} = {ans}{tokenizer.eos_token}"
            texts.append(text)
        return tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['answer', 'instruction'])
    if n_samples is not None:
        return dataset
    return dataset['train']