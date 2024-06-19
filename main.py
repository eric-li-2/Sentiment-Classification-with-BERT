from model_file import make_model
from dataset_file import create_dataloaders, load_tokenizers, load_vocab

import torch

spacy_de, spacy_en = load_tokenizers()
vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
print(f"src vocab: len({len(vocab_src)})")
print(f"tgt vocab: len({len(vocab_tgt)})")
