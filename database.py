from datasets import load_dataset

def load_and_tokenize_dataset(tokenizer, split="train", max_length=None):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length" if max_length else False, max_length=max_length)

    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.filter(lambda example: len(example["input_ids"]) > 0)
    return tokenized

