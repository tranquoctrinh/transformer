import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from transformers import AutoTokenizer
from tqdm import tqdm

from models import Transformer
from datasets import TranslateDataset

# Write funciton translate give model Transformer and source sentence and return target sentence using beam search
def translate(model, sentence, source_tokenizer, target_tokenizer, device, beam_size=5, max_len=256):
    pass

def main():
    from utils import configs
    device = torch.device(configs["device"])
    beam_size = configs["beam_size"]
    max_len = configs["target_max_seq_len"]
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
    model.load_state_dict(torch.load("./model_translate_en_vi.pt"))
    model.to(device)
    print(f"Done load model on the {device} device")  
    
    sentence = "This is my pen"
    print(translate(model, sentence, source_tokenizer, target_tokenizer, device, beam_size, max_len))

if __name__ == "__main__":
    main()