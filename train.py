"""
This file can be run with "python3 train.py" in order to train the model for 20 epochs and save the best version to model.pth
"""
import torch
import torch.nn as nn
from model_file import make_model
from dataset_file2 import get_data_loaders
import dataset_file2
from misc import DummyOptimizer, DummyScheduler
import GPUtil

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# fetch dataset
batch_size = 64
train_dl, eval_dl = get_data_loaders(batch_size)
tokenizer = dataset_file2.tokenizer
vocab_sz = len(tokenizer)

# initialize the model
model = make_model(vocab_sz, 2, N=6)

# configure the optimizer and scheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
optimizer = AdamW(model.parameters(), lr=5.0) # TODO: Tune this
def rate(step, model_size, factor, warmup): # custom scheduler used for bert
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, 512, 1, 800)
        )

# loss functions
criterion = nn.CrossEntropyLoss() # TODO: Check if label smoothing improves performance
def eval_criterion(x, tgt):
    return (x.argmax(dim=-1) == tgt).sum() / x.shape[0]

# helper function to train or eval for 1 epoch
def run_epoch(dataloader, model, optimizer, scheduler, criterion, mode="eval"):
    if mode == "train":
        model.train()
    else:
        model.eval()

    # progress_bar = tqdm(range(len(dataloader)))
    total_loss = 0
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        src_ids = batch["input_ids"]
        # src_mask = batch.attention_mask # should just be all ones
        # a mask of none implies no masking
        tgt_labels = batch["labels"]

        output_h = model.encode(src_ids, None)
        logits = model.generator(output_h[:,0].squeeze())
        loss = criterion(logits, tgt_labels)
        
        if mode == "train":
            loss.backward()
        total_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    return total_loss 



# main training loop
from tqdm.auto import tqdm

num_epochs = 20
model.to(device)
best_accuracy = 0

for epoch in range(num_epochs):
    print(f"Running Epoch {epoch+1}...")

    # progress_bar = tqdm(range(len(train_dl)))
    train_loss = 0
    model.train()
    for i, batch in tqdm(enumerate(train_dl)):
        batch = {k: v.to(device) for k, v in batch.items()}
        src_ids = batch["input_ids"]
        # src_mask = batch.attention_mask # should just be all ones
        # a mask of none implies no masking
        tgt_labels = batch["labels"]

        output_h = model.encode(src_ids, None)
        logits = model.generator(output_h[:,0].squeeze())
        loss = criterion(logits, tgt_labels)
        
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        # progress_bar.update(1)

        # if i % 400 == 0:
        #     model.eval()
        #     eval_loss = run_epoch(eval_dl, model, DummyOptimizer(), DummyScheduler(), eval_criterion, "eval")
        #     avg_accuracy = eval_loss / len(eval_dl)
        #     # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        #     # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        #     # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        #     print(f"Evalation accuracy: {avg_accuracy}", len(eval_dl))
        #     model.train()
    model.eval()
    eval_loss = run_epoch(eval_dl, model, DummyOptimizer(), DummyScheduler(), eval_criterion, "eval")
    avg_accuracy = eval_loss / len(eval_dl)
    print(f"Evalation accuracy: {avg_accuracy}", len(eval_dl))
    if avg_accuracy > best_accuracy:
        torch.save(model.state_dict(), "model.pth")
    model.train()
        

    # GPUtil.showUtilization()
    print(f"lr = {optimizer.param_groups[0]['lr']}")
    torch.cuda.empty_cache()

    print(f"training loss: {train_loss}")
print(f"Best evaluation accuracy was {best_accuracy}. You can check out the trained model by running demo.py")


