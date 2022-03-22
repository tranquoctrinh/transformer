import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from transformers import AutoTokenizer


from datasets import TranslateDataset
from models import Transformer


def train_model(model, train_loader, optim, n_epochs, target_pad, device, print_freq=100):
    model.train()
    start = time.time()
    temp = start
    total_loss = 0
    for epoch in range(n_epochs):
        for i, batch in enumerate(train_loader):
            source, source_mask, target, target_mask = batch["source_ids"].to(device), batch["source_mask"].to(device), batch["target_ids"].to(device), batch["target_mask"].to(device)
            target_input = target[:, :-1]
            preds = model(source, target_input, source_mask, target_mask)
            optim.zero_grad()
            
            targets = target[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets, ignore_index=target_pad)
            loss.backward()
            optim.step()
            
            total_loss += loss.data[0]
            if (i + 1) % print_freq == 0:
                loss_avg = total_loss / print_freq
                print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" % ((time.time() - start) // 60, epoch + 1, i + 1, loss_avg, time.time() - temp, print_freq))
                total_loss = 0
                temp = time.time()

def validate_model(model, valid_loader, device):
    model.eval()
    total_loss = 0
    for i, batch in enumerate(valid_loader):
        source, source_mask, target, target_mask = batch["source_ids"].to(device), batch["source_mask"].to(device), batch["target_ids"].to(device), batch["target_mask"].to(device)
        target_input = target[:, :-1]
        preds = model(source, target_input, source_mask, target_mask)
        targets = target[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets, ignore_index=target_pad)
        total_loss += loss.data[0]
    return total_loss / i


def main():
    # configs
    configs = {
        "train_source_data":"./data_en_vi/train.en",
        "train_target_data":"./data_en_vi/train.vi",
        "valid_source_data":"./data_en_vi/tst2013.en",
        "valid_target_data":"./data_en_vi/tst2013.vi",
        "source_tokenizer":"bert-base-uncased",
        "target_tokenizer":"vinai/phobert-base",
        "source_max_seq_len":256,
        "target_max_seq_len":256,
        "batch_size":1024,
        "device":"cpu",
        "embedding_dim": 512,
        "n_layers": 6,
        "n_heads": 8,
        "dropout": 0.1,
        "lr":0.0001,
        "epochs":30,
        "print_freq": 200,
        "k":5,
    }
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
    model = model.to

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

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
    train_model(model, train_loader, optim, configs["epochs"], configs["target_pad"], device, configs["print_freq"])
    valid_loss = validate_model(model, valid_loader, device)
    print("valid loss = %.3f" % valid_loss)
    
    # save model
    torch.save(model.state_dict(), "./model_translate_en_vi.pt")
    print("model saved")

    # load model
    # model.load_state_dict(torch.load("./model_translate_en_vi.pt"))


if __name__ == "__main__":
    main()
