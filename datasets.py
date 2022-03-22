import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import re

class TranslateDataset(Dataset):
    def __init__(self, source_tokenizer, target_tokenizer, source_data, target_data, source_max_seq_len, target_max_seq_len):
        self.source_data = source_data
        self.target_data = target_data
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_max_seq_len = source_max_seq_len
        self.target_max_seq_len = target_max_seq_len
    
    def preprocess_seq(self, seq):
        seq = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(seq))
        seq = re.sub(r"[ ]+", " ", seq)
        seq = re.sub(r"\!+", "!", seq)
        seq = re.sub(r"\,+", ",", seq)
        seq = re.sub(r"\?+", "?", seq)
        seq = seq.lower()
        return seq

    def convert_line_uncased(self, tokenizer, text, max_seq_len):
        tokens = tokenizer.tokenize(text)[:max_seq_len-2]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        tokens += [tokenizer.pad_token]*(max_seq_len - len(tokens))
        token_idx = tokenizer.convert_tokens_to_ids(tokens)
        return tokens, token_idx
    
    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, index):
        # source_seq, source_idx = self.convert_line_uncased(
        #     tokenizer=self.source_tokenizer, 
        #     text=self.preprocess_seq(self.source_data[index]), 
        #     max_seq_len=self.source_max_seq_len
        # )
        # target_seq, target_idx = self.convert_line_uncased(
        #     tokenizer=self.target_tokenizer, 
        #     text=self.preprocess_seq(self.target_data[index]), 
        #     max_seq_len=self.target_max_seq_len
        # )

        source = self.source_tokenizer(
            text=self.preprocess_seq(self.source_data[index]),
            padding="max_length", 
            max_length=self.source_max_seq_len, 
            truncation=True, 
            return_tensors="pt"
        )
        target = self.target_tokenizer(
            text=self.preprocess_seq(self.target_data[index]),
            padding="max_length",
            max_length=self.target_max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "source_seq": self.source_data[index],
            "source_ids": source["input_ids"][0],
            "source_mask": source["attention_mask"][0],
            "target_seq": self.target_data[index],
            "target_ids": target["input_ids"][0],
            "target_mask": target["attention_mask"][0]
            }

def main():
    configs = {
        "train_source_data":"./data_en_vi/train.en",
        "train_target_data":"./data_en_vi/train.vi",
        "valid_source_data":"./data_en_vi/tst2013.en",
        "valid_target_data":"./data_en_vi/tst2013.vi",
        "source_tokenizer":"bert-base-uncased",
        "target_tokenizer":"vinai/phobert-base",
        "source_max_seq_len":256,
        "target_max_seq_len":256,
        "batch_size":4,
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
        source_data = open(source_file).read().strip().split("\n")
        target_data = open(target_file).read().strip().split("\n")
        return source_data, target_data

    train_src_data, train_trg_data = read_data(configs["train_source_data"], configs["train_target_data"])
    valid_src_data, valid_trg_data = read_data(configs["valid_source_data"], configs["valid_target_data"])

    source_tokenizer = AutoTokenizer.from_pretrained(configs["source_tokenizer"])
    target_tokenizer = AutoTokenizer.from_pretrained(configs["target_tokenizer"])
    train_dataset = TranslateDataset(
        source_tokenizer=source_tokenizer, 
        target_tokenizer=target_tokenizer, 
        source_data=train_src_data, 
        target_data=train_trg_data, 
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"]
    )
    print(train_dataset[0])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=True
    )
    for batch in train_loader:
        print(batch["source_seq"])
        print(batch["target_seq"])
        print(batch["source_ids"])
        print(batch["source_mask"])
        print(batch["target_ids"])
        print(batch["target_mask"])
        break

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()