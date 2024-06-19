"""
Run this program to use a saved model to evaluate the sentiment of user input
"""
import torch
from os.path import exists
from model_file import make_model
from dataset_file2 import tokenizer

model_path = "model.pth"
if not exists(model_path):
    print("No model found. Please first run train.py")
    exit(1)
vocab_sz = len(tokenizer)
model = make_model(vocab_sz, 2, 6)
model.load_state_dict(torch.load("model.pth"))

inp = input('Input the sentence you would like to evaluate: ')
tokens = tokenizer([inp], truncation=True)
input_ids = torch.tensor(tokens.input_ids)
print(f"input ids: {input_ids}")
logits = model.generator(model.encode(input_ids, None))
logits = logits[0,0]
idx = logits.argmax(dim=-1).item()
label_map = {0: "NEGATIVE", 1: "POSITIVE"}
print(label_map[idx])


