import torch
from torch.utils.data import Dataset



class TranslateDataset(Dataset):
    def __init__(self, src_spacy, trg_spacy, source_data, target_data, max_seq_len):
        self.source_data = source_data
        self.target_data = target_data
        self.source_spacy_model = spacy.load(src_spacy) # en
        self.target_spacy_model = spacy.load(trg_spacy) # vi_spacy_model
        self.max_seq_len = max_seq_len
    
    def tokenizer(self, spacy_model, sentence):
        return [tok.text for tok in spacy_model.tokenizer(sentence)]
    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, index):
        source_seq = self.tokenizer(self.source_data[index])
        target_seq = self.tokenizer(self.target_data[index])
        return {"source": source_seq, "target": target_seq}

def main():
    import spacy
    
    def read_data(source_file, target_file):
        source_data = open(source_file).read().strip().split('\n')
        target_data = open(target_file).read().strip().split('\n')
        return source_data, target_data
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
    train_src_data, train_trg_data = read_data(prams['train_src_data'], prams['train_trg_data'])
    valid_src_data, valid_trg_data = read_data(prams['valid_src_data'], prams['valid_trg_data'])

    import ipdb; ipdb.set_trace()