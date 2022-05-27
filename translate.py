import torch
import time
from utils import configs
from evaluate import load_model_tokenizer, translate


def main():   
    import time
    # Translate a sentence
    sentence = "My family is very poor, I had to go through hard life when I was young, now I have a better life."
    print("--- English input sentence:", sentence)
    print("--- Translating...")
    device = torch.device(configs["device"])
    model, source_tokenizer, target_tokenizer = load_model_tokenizer(configs)
    st = time.time()
    trans_sen = translate(
        model=model, 
        sentence=sentence, 
        source_tokenizer=source_tokenizer, 
        target_tokenizer=target_tokenizer, 
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"], 
        beam_size=configs["beam_size"], 
        device=device
    )
    end = time.time()
    print("--- Sentences translated into Vietnamese:", trans_sen)
    print(f"--- Time: {end-st} (s)")

if __name__ == "__main__":
    main()