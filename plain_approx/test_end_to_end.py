import gpt2_approx_checkpoint_2 as gpt2_ref
import pack
import full_gpt2
import pytest
import evaluate
from transformers import AutoTokenizer, GPT2LMHeadModel,Trainer,TrainingArguments
from tqdm import tqdm
from dataclasses import dataclass

#Numpy imports
import numpy as np

#Base imports
import copy
import math

# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
# Load device
torch_device = 'cpu'

tokenizer = AutoTokenizer.from_pretrained("gpt2")
#model = AutoModelForCausalLM.from_pretrained("gpt2")

#model = AutoModelForCausalLM.from_pretrained("gpt2",pad_token_id=tokenizer.eos_token_id).to(torch_device)
#model.to(torch_device)

tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
tokenizer.pad_token = "<|pad|>"

model = GPT2LMHeadModel.from_pretrained("gpt2")
# We need to resize the embedding layer because we added the pad token.
model.resize_token_embeddings(len(tokenizer))

"""## Correctness sanity check"""
import torch
from tqdm import tqdm
from dataclasses import dataclass

# Config object to pass through net
@dataclass
class Config():
    n_layers: int
    d_model: int
    dropout: float
    n_heads: int
    d_hidden: int
    max_len: int
    vocab_size: int
    beta_start: float
    gamma_start:float
    device: str

config = Config(
    1,
    768,
    0.1,
    12,
    3072,
    128,
    len(tokenizer),
    1.5,
    100.,
    torch_device
)

#model = GPT2LMHeadModel.from_pretrained("gpt2")
weights = { k : v for k,v in model.state_dict().items()}


'''
 nopad_mask = ids != self.tokenizer.pad_token_id
            logits: torch.Tensor = self.model(ids,None).logits

        for sent_index in range(len(text)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = [
                tok
                for i, tok in enumerate(encoding.tokens(sent_index))
                if sent_nopad_mask[i] and i != 0
            ]

            # sent_ids.shape = [len(text[sent_index]) + 1]
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")
            # ids_scores.shape = [seq_len + 1]
            sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
            # log_prob.shape = [seq_len + 1]
            sent_log_probs = sent_ids_scores - sent_logits.logsumexp(1)
'''
def get_sub_mask(config):
        #x: (batch_size, seq_len)
        subsequent_mask = torch.tril(torch.ones((config.max_len, config.max_len)))
        return subsequent_mask

# def main(inputs,config,tokens_to_gen=5):
    
#     #toks = tokens['input_ids']

#     output_ids = generate(inputs,config,tokens_to_gen)

#     return output_ids

#out_toks = main("Alan Turing theorized that computers would one day become",config)
#print("Decoded output:",50*'-')
#print(tokenizer.decode(greedy_output[0],skip_special_tokens=True))
#print(out_toks)

@pytest.mark.full
def test_gpt2_correctness_check():

    torch.set_default_dtype(torch.float64)

    text = "Alan Turing theorized that computers would one day become"
    tokens = tokenizer(text,max_length=config.max_len,padding="max_length",return_tensors='pt',return_attention_mask=True)
    inputs = tokens['input_ids']
    print(f"Tokenized input: {inputs[0][:10]}")

    
    # Run torch reference implementation 
    sub_mask = get_sub_mask(config)
    gpt2 = gpt2_ref.GPT2_LM(config,weights).eval()
    exp = gpt2(inputs,inputs,sub_mask)[0][0]
    exp = exp.cpu().detach().numpy()
    print(exp.shape)
    
    # Run CKKS-ified version
    embedded,ckks_weights = full_gpt2.gpt2_setup(inputs,config,weights)
    res = full_gpt2.gpt2_inference(embedded,config,ckks_weights)

    res = pack.pack_tight(res)

    print(f"EXP: {exp[0][:10]}")
    print(f"Res: {res[0][:10]}")

    assert np.isclose(exp.reshape(-1),res.reshape(-1),atol=0.01).all()