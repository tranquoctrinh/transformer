import torch
import json
import os

configs = {
    "train_source_data":"./data_en_vi/train.en",
    "train_target_data":"./data_en_vi/train.vi",
    "valid_source_data":"./data_en_vi/tst2013.en",
    "valid_target_data":"./data_en_vi/tst2013.vi",
    "source_tokenizer":"bert-base-uncased",
    "target_tokenizer":"vinai/phobert-base",
    "source_max_seq_len":256,
    "target_max_seq_len":256,
    "batch_size":20,
    "device":"cuda:0" if torch.cuda.is_available() else "cpu",
    "embedding_dim": 512,
    "n_layers": 6,
    "n_heads": 8,
    "dropout": 0.1,
    "lr":0.0001,
    "n_epochs":50,
    "print_freq": 5,
    "beam_size":3,
    "model_path":"./model_transformer_translate_en_vi.pt",
    "early_stopping":5
}

import matplotlib.pyplot as plt

# visualize log
def plot_loss(log_path, log_dir):
    log = json.load(open(log_path, "r"))

    plt.figure()
    plt.plot(log["train_loss"], label="train loss")
    plt.plot(log["valid_loss"], label="valid loss")
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss_epoch.png"))

    # plot batch loss
    plt.figure()
    lst = log["train_batch_loss"]
    n = int(len(log["train_batch_loss"]) / len(log["valid_batch_loss"]))
    train_batch_loss = [lst[i:i + n][0] for i in range(0, len(lst), n)]
    plt.plot(train_batch_loss, label="train loss")
    plt.plot(log["valid_batch_loss"], label="valid loss")
    plt.title("Loss per batch")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss_batch.png"))