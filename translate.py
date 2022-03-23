import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from transformers import AutoTokenizer
from tqdm import tqdm

from models import Transformer

# This funciton will translate give a source sentence and return target sentence using beam search
def translate(model, source_sentence, source_tokenizer, target_tokenizer, target_max_seq_len=256, beam_size=5, device=torch.device("cpu")):
    # Convert source sentence to tensor
    source_tokens = source_tokenizer.encode(source_sentence)
    source_tensor = torch.tensor(source_tokens).unsqueeze(0)
    # Add batch dimension
    source_tensor = source_tensor.to(device)
    # Create source sentence mask
    source_mask = model.make_source_mask(source_tensor, source_tokenizer.pad_token_id).to(device)
    # Initialize decoder hidden state
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
            pred = F.softmax(pred, dim=-1).view(-1)
            # Get top k tokens
            top_k_scores, top_k_tokens = pred.topk(beam_size)
            # Update beams
            for i in range(beam_size):
                new_beams.append((beam[0] + [top_k_tokens[i].item()], beam[1] + top_k_scores[i].item()))
        
        import copy
        beams = copy.deepcopy(new_beams)
        # sort beams by score
        beams = sorted(beams, key=lambda x: x[1], reverse=True)[:beam_size]
        # add completed beams to completed list and reduce beam size
        for beam in beams:
            if beam[0][-1] == target_tokenizer.eos_token_id:
                completed.append(beam)
                beams.remove(beam)
                beam_size -= 1
        
        # print screen progress
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
    target_sentence = target_tokenizer.decode(target_tokens)
    return target_sentence


def main():
    from utils import configs
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
    model.load_state_dict(torch.load("./model_translate_en_vi.pt"))
    model.eval()
    model.to(device)
    print(f"Done load model on the {device} device")  
    
    # Translate a sentence
    sentence = "This is my pen"
    print(translate(model, sentence, source_tokenizer, target_tokenizer, configs["target_max_seq_len"], configs["beam_size"], device))


if __name__ == "__main__":
    main()