import sys
import math
import warnings
import itertools
import pyterrier as pt
import pandas as pd
from collections import defaultdict
from pyterrier.model import add_ranks
import torch
from torch.nn import functional as F
from pyterrier.transformer import Transformer
from typing import List
import re

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig


def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path, 
        num_labels=1,
        torch_dtype=torch.float16  # Use FP16 to save memory
    )
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model




class RankLammaReRanker(Transformer):
    def __init__(self, 
                 tok_model='meta-llama/Llama-2-7b-hf',
                 model='castorini/rankllama-v1-7b-lora-passage',
                 batch_size=4,
                 text_field='text',
                 verbose=False,
                 device=None):
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        if isinstance(self.device, torch.device) and self.device.type == 'cuda' and self.device.index is not None:
            self.device = torch.device(f'cuda:{self.device.index}')
        self.tokenizer = AutoTokenizer.from_pretrained(tok_model)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model_name = model
        self.model = get_model(model)
        self.model.to(self.device)
        self.model.eval()
        self.text_field = text_field

    def __str__(self):
        return f"RankLamma({self.model_name})"

    def transform(self, run):
        scores = []
        queries, texts = run['query'], run[self.text_field]
        it = range(0, len(queries), self.batch_size)

        if self.verbose:
            it = pt.tqdm(it, desc='RankLamma', unit='batches')
            
        # for start_idx in it:
        #     rng = slice(start_idx, start_idx+self.batch_size) # same as start_idx:start_idx+self.batch_size

        for query, passage in zip(queries, texts):

            # expanded_q = [f'query: {q}' for q in queries[rng]]
            # expanded_docs = [f'document: {d}' for d in texts[rng]]
            #enc = self.tokenizer.batch_encode_plus([f'query: {q} document: {d}' for q, d in zip(queries[rng], texts[rng])], return_tensors='pt', padding='longest')
            enc = self.tokenizer(f'query: {query}', f'document: {passage}', return_tensors='pt', max_length=4096, truncation=True)
            # enc = self.tokenizer.batch_encode_plus(list(zip(expanded_q, expanded_docs)), return_tensors='pt', padding='longest')
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                result = self.model(**enc).logits
                batch_scores = result[:, 0].cpu().tolist()
                scores.extend(batch_scores)
            
            # Clear GPU cache after each batch
            del enc, result
            torch.cuda.empty_cache()

                
        run = run.drop(columns=['score', 'rank'], errors='ignore').assign(score=scores)
        run = add_ranks(run)
        return run
