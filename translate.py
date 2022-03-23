import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from transformers import AutoTokenizer
from tqdm import tqdm

from models import Transformer
from datasets import TranslateDataset

# This funciton will translate give a source sentence and return target sentence using beam search
def translate(model, source_sentence, source_tokenizer, target_tokenizer, target_max_seq_len=256, beam_size=5, device=torch.device("cpu")):
    # Convert source sentence to tensor
    source_tokens = source_tokenizer.encode(source_sentence)
    source_tensor = torch.tensor(source_tokens).unsqueeze(0)
    # Add batch dimension
    source_tensor = source_tensor.to(device)
    # Create source sentence mask
    # source_mask = model.make_source_mask(source_tensor)
    source_mask = torch.ones(1, len(source_tokens)).to(device)
    # Initialize decoder hidden state
    decoder_hidden = model.encoder.forward(source_tensor, source_mask)
    # Initialize decoder memory cell
    decoder_cell = torch.zeros(1, 1, 1024).to(device)
    # Initialize beam list
    beams = [([target_tokenizer.bos_token_id], 0, decoder_hidden, decoder_cell)]
    # Start decoding
    for _ in range(target_max_seq_len):
        # Get input token
        input_token = torch.tensor([[beams[0][0][-1]]]).to(device)
        # Get hidden and cell states
        hidden, cell = beams[0][2], beams[0][3]
        # Create mask
        target_mask = model.make_target_mask(input_token)
        # Decoder forward pass
        pred, hidden, cell = model.decoder.forward(input_token, target_mask, hidden, cell)
        # Get top k tokens
        top_k_tokens = torch.argsort(pred[0], descending=True)[:beam_size]
        # Update beams
        beams = [(beams[0][0] + [int(top_k_tokens[0])], pred[0][top_k_tokens[0]], hidden, cell)]
        for i in range(1, beam_size):
            beams.append((beams[0][0] + [int(top_k_tokens[i])], pred[0][top_k_tokens[i]], hidden, cell))
        # Sort the beams
        beams.sort(key=lambda x: x[1], reverse=True)
        # Get rid of the worst beam
        beams = beams[:beam_size]
    # Get target sentence tokens
    target_sentence_tokens = beams[0][0][1:]
    # Convert target sentence from tokens to string
    target_sentence = target_tokenizer.decode(target_sentence_tokens)
    return target_sentence


def main():
    from utils import configs
    device = torch.device(configs["device"])
    device = torch.device("cpu")
    source_tokenizer = AutoTokenizer.from_pretrained(configs["source_tokenizer"])
    target_tokenizer = AutoTokenizer.from_pretrained(configs["target_tokenizer"])  

    # Load model Transformer
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
    model.eval()
    model.to(device)
    print(f"Done load model on the {device} device")  
    
    # Translate a sentence
    sentence = "This is my pen"
    print(translate(model, sentence, source_tokenizer, target_tokenizer, device, configs["target_max_seq_len"], configs["beam_size"], device))

if __name__ == "__main__":
    main()