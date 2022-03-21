import torch
import torch.nn as nn
import time

from .model import Transformer


def train_model(model, optim, n_epochs, print_every=100):
    model.train()
    start = time.time()
    temp = start
    total_loss = 0
    for epoch in range(n_epochs):
        for i, batch in enumerate(train_iter):
            src = batch.English.transpose(0,1)
            trg = batch.French.transpose(0,1)
            trg_input = trg[:, :-1]
            targets = trg[:, 1:].contiguous().view(-1)
            src_mask, trg_mask = create_masks(src, trg_input)
            preds = model(src, trg_input, src_mask, trg_mask)
            optim.zero_grad()
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), results, ignore_index=target_pad)
            loss.backward()
            optim.step()
            
            total_loss += loss.data[0]
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f, %ds per %d iters" % ((time.time() - start) // 60, epoch + 1, i + 1, loss_avg, time.time() - temp, print_every))
                total_loss = 0
                temp = time.time()

def main():
    embedding_dim = 512
    heads = 8
    N = 6
    src_vocab = len(EN_TEXT.vocab)
    trg_vocab = len(FR_TEXT.vocab)

    model = Transformer(src_vocab, trg_vocab, embedding_dim, N, heads)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


if __name__ == "__main__":
    main()
