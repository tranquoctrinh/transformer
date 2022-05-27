import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import re
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from torchtext.data.metrics import bleu_score

from utils import configs
from models import Transformer
from datasets import TranslateDataset
from train import read_data
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction()


def load_model_tokenizer(configs):
    """
    This function will load model and tokenizer from pretrained model and tokenizer
    """
    device = torch.device(configs["device"])
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
    model.load_state_dict(torch.load(configs["model_path"]))
    model.eval()
    model.to(device)
    print(f"Done load model on the {device} device")  
    return model, source_tokenizer, target_tokenizer


def translate(model, sentence, source_tokenizer, target_tokenizer, source_max_seq_len=256, 
    target_max_seq_len=256, beam_size=3, device=torch.device("cpu"), print_process=False):
    """
    This funciton will translate give a source sentence and return target sentence using beam search
    """
    # Convert source sentence to tensor
    source_tokens = source_tokenizer.encode(sentence)[:source_max_seq_len]
    source_tensor = torch.tensor(source_tokens).unsqueeze(0).to(device)
    # Create source sentence mask
    source_mask = model.make_source_mask(source_tensor, source_tokenizer.pad_token_id).to(device)
    # Feed forward Encoder
    encoder_output = model.encoder.forward(source_tensor, source_mask)
    # Initialize beam list
    beams = [([target_tokenizer.bos_token_id], 0)]
    completed = []
    # Start decoding
    for _ in range(target_max_seq_len):
        new_beams = []
        for beam in beams:
            # Get input token
            input_token = torch.tensor([beam[0]]).to(device)
            # Create mask
            target_mask = model.make_target_mask(input_token).to(device)
            # Decoder forward pass
            pred = model.decoder.forward(input_token, encoder_output, source_mask, target_mask)
            # Forward to linear classify token in vocab and Softmax
            pred = F.softmax(model.final_linear(pred), dim=-1)
            # Get tail predict token
            pred = pred[:, -1, :].view(-1)
            # Get top k tokens
            top_k_scores, top_k_tokens = pred.topk(beam_size)
            # Update beams
            for i in range(beam_size):
                new_beams.append((beam[0] + [top_k_tokens[i].item()], beam[1] + top_k_scores[i].item()))
        
        import copy
        beams = copy.deepcopy(new_beams)
        # Sort beams by score
        beams = sorted(beams, key=lambda x: x[1], reverse=True)[:beam_size]
        # Add completed beams to completed list and reduce beam size
        for beam in beams:
            if beam[0][-1] == target_tokenizer.eos_token_id:
                completed.append(beam)
                beams.remove(beam)
                beam_size -= 1
        
        # Print screen progress
        if print_process:
            print(f"Step {_+1}/{target_max_seq_len}")
            print(f"Beam size: {beam_size}")
            print(f"Beams: {[target_tokenizer.decode(beam[0]) for beam in beams]}")
            print(f"Completed beams: {[target_tokenizer.decode(beam[0]) for beam in completed]}")
            print(f"Beams score: {[beam[1] for beam in beams]}")
            print("-"*100)

        if beam_size == 0:
            break


    # Sort the completed beams
    completed.sort(key=lambda x: x[1], reverse=True)
    # Get target sentence tokens
    target_tokens = completed[0][0]
    # Convert target sentence from tokens to string
    target_sentence = target_tokenizer.decode(target_tokens, skip_special_tokens=True)
    return target_sentence
    

def calculate_bleu_score(model, source_tokenizer, target_tokenizer, configs):
    device = torch.device(configs["device"])
    def preprocess_seq(seq):
        seq = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(seq))
        seq = re.sub(r"[ ]+", " ", seq)
        seq = re.sub(r"\!+", "!", seq)
        seq = re.sub(r"\,+", ",", seq)
        seq = re.sub(r"\?+", "?", seq)
        seq = seq.lower()
        return seq
    
    valid_src_data, valid_trg_data = read_data(configs["valid_source_data"], configs["valid_target_data"])

    pred_sents = []
    for sentence in tqdm(valid_src_data):
        pred_trg = translate(model, sentence, source_tokenizer, target_tokenizer, configs["source_max_seq_len"], configs["target_max_seq_len"], configs["beam_size"], device)
        pred_sents.append(pred_trg)
    
    # write prediction to file
    with open("logs/predict_valid.txt", "wb") as f:
        for sent in pred_sents:
            f.write(f"{sent}\n")

    hypotheses = [preprocess_seq(sent).split() for sent in pred_sents]
    references = [[sent.split()] for sent in valid_trg_data]
    
    weights = [(0.5, 0.5),(0.333, 0.333, 0.334),(0.25, 0.25, 0.25, 0.25)]
    bleu_2 = corpus_bleu(references, hypotheses, weights=weights[0])
    bleu_3 = corpus_bleu(references, hypotheses, weights=weights[1])
    bleu_4 = corpus_bleu(references, hypotheses, weights=weights[2])
    print(f"BLEU-2: {bleu_2} | BLEU-3: {bleu_3} | BLEU-4: {bleu_4}")
    return {"bleu_2": bleu_2, "bleu_3": bleu_3, "bleu_4": bleu_4}


def main():
    model, source_tokenizer, target_tokenizer = load_model_tokenizer(configs)
    bleus = calculate_bleu_score(model, source_tokenizer, target_tokenizer, configs)


if __name__ == "__main__":
    main()