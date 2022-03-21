import torch
import torch.nn as nn
import time
import torch.nn.function as F


from .model import Transformer


def train_model(model, train_loader, optim, n_epochs, device, print_freq=100):
    model.train()
    start = time.time()
    temp = start
    total_loss = 0
    for epoch in range(n_epochs):
        for i, batch in enumerate(train_loader):
            source, target = batch["source"].to(device), batch["target"].to(device)
            target_input = target[:, :-1]
            targets = target[:, 1:].contiguous().view(-1)
            source_mask, target_mask = create_masks(source, target_input)
            preds = model(source, target_input, source_mask, target_mask)
            optim.zero_grad()
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), results, ignore_index=target_pad)
            loss.backward()
            optim.step()
            
            total_loss += loss.data[0]
            if (i + 1) % print_freq == 0:
                loss_avg = total_loss / print_freq
                print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" % ((time.time() - start) // 60, epoch + 1, i + 1, loss_avg, time.time() - temp, print_freq))
                total_loss = 0
                temp = time.time()

def main():
    # parameters
    prams = {
        "train_source_data":"./data_en_vi/train.en",
        "train_target_data":"./data_en_vi/train.vi",
        "valid_source_data":"./data_en_vi/tst2013.en",
        "valid_target_data":"./data_en_vi/tst2013.vi",
        "source_lang":"en",
        "target_lang":"vi_spacy_model",
        "max_strlen":160,
        "batchsize":1500,
        "device":"cuda:0",
        "embedding_dim": 512,
        "n_layers": 6,
        "n_heads": 8,
        "dropout": 0.1,
        "lr":0.0001,
        "epochs":30,
        "printevery": 200,
        "k":5,
    }
    def read_data(source_file, target_file):
        source_data = open(source_file).read().strip().split('\n')
        target_data = open(target_file).read().strip().split('\n')
        return source_data, target_data

    train_src_data, train_trg_data = read_data(prams['train_src_data'], prams['train_trg_data'])
    valid_src_data, valid_trg_data = read_data(prams['valid_src_data'], prams['valid_trg_data'])


    source_vocab_size = len(EN_TEXT.vocab)
    target_vocab_size = len(VN_TEXT.vocab)

    model = Transformer(
        source_vocab_size, 
        target_vocab_size, 
        prams['embedding_dim'], 
        prams["n_layers"], 
        prams["n_heads"], 
        prams["dropout"]
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


if __name__ == "__main__":
    main()
