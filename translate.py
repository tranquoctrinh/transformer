import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from transformers import AutoTokenizer
from tqdm import tqdm

from models import Transformer
from datasets import TranslateDataset

# Write funciton translate give model Transformer and source sentence and return target sentence using beam search
def translate(model, source, source_pad_id, target_pad_id, device, max_len=100):
