"""
This file loads a dataset and tokenizer
"""
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

# Load the dataset and tokenizer
raw_datasets = load_dataset("glue", "sst2")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# print out some samples
# print(raw_datasets)
# print(raw_datasets["train"][4])
# print(raw_datasets["train"].features["label"])

# tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # used for dynamic padding

tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
vocab_sz = len(tokenizer)
# print(tokenized_datasets["train"][0])

from torch.utils.data import DataLoader

def get_data_loaders(batch_size=8):
    # truncate the eval dataset to speed things up
    train_ds = tokenized_datasets["train"].shuffle(seed=10) # .select()
    eval_ds = tokenized_datasets["validation"].shuffle(seed=10).select(range(200))

    # put hf dataset into pytorch dataloader
    train_dataloader = DataLoader(
        train_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        eval_ds, batch_size=batch_size, collate_fn=data_collator
    )
    # for batch in train_dataloader:
    #     break
    # print({k: v.shape for k, v in batch.items()})
    return train_dataloader, eval_dataloader