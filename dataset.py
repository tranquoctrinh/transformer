import torch
from torch.utils.data import Dataset
import spacy
import re

class TranslateDataset(Dataset):
    def __init__(self, src_spacy, trg_spacy, source_data, target_data, max_seq_len):
        self.source_data = source_data
        self.target_data = target_data
        self.source_spacy_model = spacy.load(src_spacy)
        self.target_spacy_model = spacy.load(trg_spacy)
        self.max_seq_len = max_seq_len
    
    def tokenizer(self, spacy_model, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        tokens = [tok.text for tok in spacy_model.tokenizer(sentence) if tok.text]
        token_idx = [tok.orth for tok in spacy_model.tokenizer(sentence) if tok.text]
        return token_idx
    
    def convert_idx_to_token(self, spacy_model, idx_list):
        token_list = []
        for idx in idx_list:
            token_list.append(spacy_model.vocab[idx].text)
        return token_list
    
    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, index):
        source_seq = self.tokenizer(self.source_spacy_model, self.source_data[index])
        target_seq = self.tokenizer(self.target_spacy_model, self.target_data[index])
        import ipdb; ipdb.set_trace()
        return {"source_tokens": source_seq, "target_tokens": target_seq}

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
        "source_lang":"en_core_web_sm",
        "target_lang":"vi_core_news_lg",
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
    train_src_data, train_trg_data = read_data(prams['train_source_data'], prams['train_target_data'])
    valid_src_data, valid_trg_data = read_data(prams['valid_source_data'], prams['valid_target_data'])
    train_dataset = TranslateDataset(prams['source_lang'], prams['target_lang'], train_src_data, train_trg_data, prams['max_strlen'])
    print(train_dataset[0])
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()