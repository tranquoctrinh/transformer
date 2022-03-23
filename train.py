import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from datetime import datetime, timedelta

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import TranslateDataset
from models import Transformer


def validate_model(model, valid_loader, source_pad_id, target_pad_id, device):
    model.eval()
    total_loss = []
    bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Validating")
    for i, batch in bar:
        source, target = batch["source_ids"].to(device), batch["target_ids"].to(device)
        target_input = target[:, :-1]
        source_mask, target_mask = model.make_source_mask(source, source_pad_id), model.make_target_mask(target_input)
        preds = model(source, target_input, source_mask, target_mask)
        gold = target[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), gold, ignore_index=target_pad_id)
        total_loss.append(loss.item())
    return sum(total_loss) / len(total_loss)


def train_model(model, train_loader, valid_loader, optim, n_epochs, source_pad_id, target_pad_id, device, print_freq=100):
    best_val_loss = np.Inf
    model.train()
    total_loss = []
    for epoch in range(n_epochs):
        bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training epoch {epoch+1}/{n_epochs}")
        for i, batch in bar:
            source, target = batch["source_ids"].to(device), batch["target_ids"].to(device)
            target_input = target[:, :-1]
            source_mask, target_mask = model.make_source_mask(source, source_pad_id), model.make_target_mask(target_input)
            preds = model(source, target_input, source_mask, target_mask)
            optim.zero_grad()
            gold = target[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), gold, ignore_index=target_pad_id)
            loss.backward()
            optim.step()
            total_loss.append(loss.item())
            bar.set_postfix(loss=sum(total_loss) / len(total_loss))
        
        valid_loss = validate_model(
            model=model,
            valid_loader=valid_loader,
            source_pad_id=source_pad_id,
            target_pad_id=target_pad_id,
            device=device
        )
        print("------- Validate: epoch %d, valid loss = %.3f" % (epoch + 1, valid_loss))
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            # save model
            torch.save(model.state_dict(), "./model_translate_en_vi.pt")
            print("Detect improment and save the best model")
        torch.cuda.empty_cache()



def main():
    # configs
    from utils import configs
    
    def read_data(source_file, target_file):
        source_data = open(source_file).read().strip().split("\n")
        target_data = open(target_file).read().strip().split("\n")
        return source_data, target_data

    train_src_data, train_trg_data = read_data(configs["train_source_data"], configs["train_target_data"])
    valid_src_data, valid_trg_data = read_data(configs["valid_source_data"], configs["valid_target_data"])
    source_tokenizer = AutoTokenizer.from_pretrained(configs["source_tokenizer"])
    target_tokenizer = AutoTokenizer.from_pretrained(configs["target_tokenizer"])

    model = Transformer(
        source_vocab_size=source_tokenizer.vocab_size,
        target_vocab_size=target_tokenizer.vocab_size,
        embedding_dim=configs["embedding_dim"],
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
        num_layers=configs["n_layers"],
        num_heads=configs["n_heads"],
        dropout=configs["dropout"]
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    optim = torch.optim.Adam(model.parameters(), lr=configs["lr"], betas=(0.9, 0.98), eps=1e-9)

    train_dataset = TranslateDataset(
        source_tokenizer=source_tokenizer, 
        target_tokenizer=target_tokenizer, 
        source_data=train_src_data, 
        target_data=train_trg_data, 
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
    )
    valid_dataset = TranslateDataset(
        source_tokenizer=source_tokenizer, 
        target_tokenizer=target_tokenizer, 
        source_data=valid_src_data, 
        target_data=valid_trg_data, 
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
    )

    device = torch.device(configs["device"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=configs["batch_size"],
        shuffle=False
    )

    model.to(configs["device"])
    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optim=optim,
        n_epochs=configs["n_epochs"],
        source_pad_id=source_tokenizer.pad_token_id,
        target_pad_id=target_tokenizer.pad_token_id,
        device=device,
        print_freq=configs["print_freq"]
    )

if __name__ == "__main__":
    main()
