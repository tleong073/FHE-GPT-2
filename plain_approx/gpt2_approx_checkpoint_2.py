# -*- coding: utf-8 -*-
#Base imports
import copy
import math

# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F


"""## GPT-2 Approximation

### Embedding Layer
"""
torch.set_default_dtype(torch.float64)


class EmbeddingLayer(torch.nn.Module):
    def __init__(self,config,weights):
        super().__init__()
        self.token = nn.Embedding(config.vocab_size,config.d_model,padding_idx=0)
        self.token.weight = nn.Parameter(weights['transformer.wte.weight']) # pretrained token embedding matrix

        self.pe = nn.Parameter(weights['transformer.wpe.weight']) # positional encodings

    def forward(self,X):
      x = self.token(X) + self.pe[:X.shape[1]] # (batch,d_model)
      #print(f"Embedding out: {x[0][0][:10]}")
      return x

"""### Multi-Headed Attention

### Softmax approx
"""

class ApproxSoftmax(nn.Module):
  def __init__(self,config,weights,idx):
    super().__init__()
    #print(config.beta_start,config.gamma_start)
    self.betas  = nn.Parameter(torch.tensor(weights[f"consmax.{idx}.beta"]))
    self.gammas = nn.Parameter(torch.tensor(weights[f"consmax.{idx}.gamma"]))

    self.betas.retain_grad()
    self.gammas.retain_grad()

    self.config = config
    self.idx = idx

  def forward(self,X):

    toks = X.shape[2]

    betas_reshaped = self.betas.repeat(1,toks*toks,1).T.view(self.config.n_heads,toks,toks)
    gammas_reshaped = self.gammas.repeat(1,toks*toks,1).T.view(self.config.n_heads,toks,toks)

    X = X - betas_reshaped
    X = torch.exp(X)
    X = X/gammas_reshaped
    #print(X)

    #print(X.view(X.shape[0],self.config.n_heads,-1).shape)
    #X = X.view(X.shape[0],self.config.n_heads,-1)
    #max_x = torch.max(X,keepdim=True,dim=-1)[0]
    #X = X - max_x#self.betas #
    #print(torch.max(X,keepdim=True,dim=-1)[0])
    #X = torch.softmax(X,dim=-1)
    #X = torch.exp(X) / torch.logsumexp(X,keepdim=True,dim=-1)
    #print("AFTER EXP: ",torch.max(X))
    #print("After exp",X,self.gamma,X.isnan().any())
    #X = X / self.gammasx = torch.randn(5, 10)


    #print(f" Post smax Idx: {self.idx}: {torch.max(X)} {torch.max(torch.sum(X,keepdim=True,dim=-1))}")

    #print(X.shape)
    #print("AFTER: ",torch.max(X))
    #print("After gamma",X.isnan().any())
    #print(f"Attn Idx: {self.idx}: {torch.max(X)}")
    #X = X.view(X.shape[0],self.config.n_heads,toks,toks)
    return X

"""### Attention"""

def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value,smax,mask=None, dropout=None):
    'compute scaled dot product attention'
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1)) \
                            / math.sqrt(d_k) # QK^T/sqrt(d_k). Note since first dim is batch, we need to transpose last two
    #print(f"score_dim:{scores.shape} mask_dim:{mask.shape}")
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e10)

    #print(f"score_dim_after_fill:{scores.shape}")

    p_attn = torch.softmax(scores,dim=-1)
    #p_attn = smax(scores)

    #print(f"softmax out: {p_attn[0][0][0][:10]}")
    #print(f"score_dim2:{p_attn.shape}")
    return torch.matmul(p_attn,value), p_attn

'''
Expect following mask for decoder attention
causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
'''
class MultiHeadAttention(nn.Module):
    def __init__(self,config,weights,idx,dropout=0.1):
        super(MultiHeadAttention,self).__init__()

        self.d_k = config.d_model // config.n_heads # dimension for each head
        self.heads = config.n_heads # number of attention heads

        # Pre-trained weights are packed together
        qw, kw, vw = torch.split(
            torch.tensor(weights[
                f"transformer.h.{idx}.attn.c_attn.weight"
                ]), 768, dim=-1)

        qb, kb, vb = torch.split(
            weights[f"transformer.h.{idx}.attn.c_attn.bias"
            ], 768,dim=-1)

        self.linears = clones(torch.nn.Linear(config.d_model,config.d_model),4)

        with torch.no_grad():
          self.linears[0].weight = nn.Parameter(qw.T) # W_q
          self.linears[0].bias = nn.Parameter(qb) # q_bias

          self.linears[1].weight = nn.Parameter(kw.T) # W_k
          self.linears[1].bias = nn.Parameter(kb) # k_bias

          self.linears[2].weight = nn.Parameter(vw.T) # W_v
          self.linears[2].bias = nn.Parameter(vb)

          self.linears[3].weight = nn.Parameter(weights[f"transformer.h.{idx}.attn.c_proj.weight"].T)
          self.linears[3].bias = nn.Parameter(weights[f"transformer.h.{idx}.attn.c_proj.bias"])

        #self.smax = ApproxSoftmax(config,weights,idx)
        self.smax = None

        self.dropout = nn.Dropout(dropout)

    def forward(self,query,key,value,mask=None):

        nbatches = query.size(0)
        #print(f"q:{query.shape}key:{key.shape} val:{value.shape}\n")

        #tmp =  key @ self.linears[0].weight +  self.linears[0].bias
        #print(f"KW: {tmp[0][0][:10]}")

        query,key,value = \
        [l(x).view(nbatches,-1,self.heads,self.d_k).transpose(1,2) # Ensure last two dims of Q,K,V are T x d/k
            for l, x in zip(self.linears, (query,key,value))]

        #print(f"WK: {key[0][0][0][:10]}")
        #print(f"WQ: {query[0][0][0][:10]}")
        #print(f"WV: {value[0][0][0][:10]}")

        attn,_ = attention(query,key,value,self.smax,mask,self.dropout)
        #print(f"attn out: {attn[0][0][0][:10]}")

        #print(f"softmax output: {attn[0][0][0][:20]}")

        # Concat output for feedforward layer
        x = attn.transpose(1,2).contiguous().view(nbatches,-1,self.heads * self.d_k)
        #print(f"grouped output attention: {x[0][0][:20]}")


        x = self.linears[3](x)
        #print(f"Attn out: {x[0][0][:10]}")
        return x

"""### Decoder Blocks"""

class FeedForward(torch.nn.Module):
    def __init__(self,config,weights,idx):

        super(FeedForward,self).__init__()


        self.linear1 = nn.Linear(config.d_model,config.d_hidden)
        self.linear2 = nn.Linear(config.d_hidden,config.d_model)

        with torch.no_grad():
          self.linear1.weight = nn.Parameter(weights[f"transformer.h.{idx}.mlp.c_fc.weight"].T)
          self.linear1.bias = nn.Parameter(weights[f"transformer.h.{idx}.mlp.c_fc.bias"])

          self.linear2.weight = nn.Parameter(weights[f"transformer.h.{idx}.mlp.c_proj.weight"].T)
          self.linear2.bias = nn.Parameter(weights[f"transformer.h.{idx}.mlp.c_proj.bias"])


        self.activation = nn.GELU(approximate='tanh')

    def forward(self,x):
        #print(x.shape, self.linear1.weight.shape)
        x = self.linear1(x)
        #print(f"ff lin1: {x[0][0][:10]}")
        x = self.activation(x)
        #print(f"ff gelu: {x[0][0][:10]}")
        x = self.linear2(x)
        #print(f"ff lin 2 :{x[0][0][:10]}")
        return x

class DecoderLayer(nn.Module):
  def __init__(self,config,weights,idx,dropout=0.1):
    super(DecoderLayer,self).__init__()

    # Init attention layer
    self.mha = MultiHeadAttention(config,weights,idx)

    # Init LayerNorms
    self.ln_1 = nn.LayerNorm(config.d_model)
    self.ln_2 = nn.LayerNorm(config.d_model)

    with torch.no_grad():
      self.ln_1.weight = nn.Parameter(weights[f"transformer.h.{idx}.ln_1.weight"])
      self.ln_1.bias = nn.Parameter(weights[f"transformer.h.{idx}.ln_1.bias"])


      self.ln_2.weight = nn.Parameter(weights[f"transformer.h.{idx}.ln_2.weight"])
      self.ln_2.bias = nn.Parameter(weights[f"transformer.h.{idx}.ln_2.bias"])


    self.ffn = FeedForward(config,weights,idx)

    self.dropout = nn.Dropout(dropout)

  def forward(self,X,mask):
    # multi-head causal self attention
    #print(f"layer_norm: {layer_norm(x, **ln_1)[0][:10]}")
    #print(f"mha: {mha(layer_norm(x, **ln_1), **attn, n_head=n_head)[0][:10]}")
    print(f"INPUT: {X[0][0][:10]}")
    x = self.ln_1(X)
    print(f"Post ln_1: {x[0][0][:10]}")
    attn_res = self.mha(x,x,x,mask)
    print(f"Post attn: {attn_res[0][0][:10]}")
    print(f"INPUT: {X[0][0][:10] + self.dropout(attn_res)[0][0][:10]}")
    X = X + self.dropout(attn_res)  # [n_seq, n_embd] -> [n_seq, n_embd]
    print(f"Post residual1: {X[0][0][:10]}")
    x = self.ln_2(X)
    print(f"Post ln_2: {x[0][0][:10]}")
    # position-wise feed forward network
    ffn_out = self.ffn(x)
    print(f"Post FFN: {ffn_out[0][0][:10]}")
    x = X + self.dropout(ffn_out)  # [n_seq, n_embd] -> [n_seq, n_embd]
    print(f"Resid 2 out : {x[0][0][:10]}")
    return x

"""### Full Models"""

class GPT2_Approx(nn.Module):
  def __init__(self,config,weights):
    super(GPT2_Approx,self).__init__()

    self.embed = EmbeddingLayer(config,weights)

    self.decoders = nn.Sequential(*[
        DecoderLayer(config,weights,i) for i in range(config.n_layers)
    ])

    self.out = nn.LayerNorm(config.d_model)
    with torch.no_grad():
      self.out.weight = nn.Parameter(weights["transformer.ln_f.weight"])
      self.out.bias = nn.Parameter(weights["transformer.ln_f.bias"])
    self.device = config.device


  def forward(self,X,mask):
    # Embed text
    X = self.embed(X)

    #print(mask)
    for layer in self.decoders:
      X = layer(X,mask)
    #print(f"pre ln: {X[0][0][:10]}")
    x = self.out(X)
    #print(f"post ln: {x[0][0][:10]}")
    return x

from dataclasses import dataclass

@dataclass
class GPT2Output(nn.Module):
  loss: torch.tensor
  logits: torch.tensor

class GPT2_LM(nn.Module):
  def __init__(self,config,weights):
    super(GPT2_LM,self).__init__()
    self.gpt2_base = GPT2_Approx(config,weights)

    self.lm_head = nn.Linear(config.d_model,config.vocab_size,False)
    with torch.no_grad():
      self.lm_head.weight = nn.Parameter(weights["lm_head.weight"])

    self.loss_fn = nn.CrossEntropyLoss()

    self.config = config

  def forward(self,input_ids,labels,mask):
    X = self.gpt2_base(input_ids,mask)
    model_out = X
    #print(X.shape)
    # loss, logits
    X = self.lm_head(X)
    loss = None
    if labels is not None:
      # Shift so that tokens < n predict n
      shift_labels = input_ids[..., 1:].contiguous()
      shift_logits = X[..., :-1, :].contiguous()
      # Calculate per-token loss
      loss_fct = torch.nn.CrossEntropyLoss()
      #print(shift_logits.isnan().any())
      loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Flatten input to(Batch * max_len,vocab_size), target to (Batch*max_len)
    #print(shift_logits.shape,shift_labels.shape)
    return model_out,GPT2Output(loss, X)

"""### Pretrained Model and Tokenizer setup"""

import evaluate
from transformers import AutoTokenizer, GPT2LMHeadModel,Trainer,TrainingArguments
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

torch_device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained("gpt2")

config = Config(
    12,
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


# Load device

#model = AutoModelForCausalLM.from_pretrained("gpt2")

#model = AutoModelForCausalLM.from_pretrained("gpt2",pad_token_id=tokenizer.eos_token_id).to(torch_device)
#model.to(torch_device)

tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
tokenizer.pad_token = "<|pad|>"

model = GPT2LMHeadModel.from_pretrained("gpt2").to(config.device)
# We need to resize the embedding layer because we added the pad token.
model.resize_token_embeddings(len(tokenizer))

'''
for k,v in model.state_dict().items():
  print(k, v.shape)
'''

"""## Correctness sanity check"""



#model = GPT2LMHeadModel.from_pretrained("gpt2")
weights = { k : v.type(torch.float64) for k,v in model.state_dict().items()}

gpt2 = GPT2_LM(config,weights).to(config.device)


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
        subsequent_mask = torch.tril(torch.ones((config.max_len, config.max_len))).to(config.device)
        return subsequent_mask
def generate(original,config,tokens_to_gen):
    text = original
    text_len = len(original.split(" "))
    for _ in tqdm(range(tokens_to_gen), "generating"):
        #print('\nbefore',i,"asdas")
        toks = tokenizer(text,max_length=config.max_len,padding='max_length',return_tensors='pt',return_attention_mask=True)
        inputs = toks['input_ids'].to(config.device)
        print(inputs[0][:10])
        sub_mask = get_sub_mask(config)
        outputs= gpt2.forward(inputs,inputs,sub_mask)
        print(outputs.logits[-1][-1].shape,len(text))
        next_id = torch.argmax(outputs.logits[-1][text_len][:-1])
        text += tokenizer.decode(next_id)
        text_len+=1
        print("Progress: ",text)
        #print(inputs[0][:20])
        #inputs.append(int(next_id))
    return text

def main(inputs,config,tokens_to_gen=5):
    
    #toks = tokens['input_ids']

    output_ids = generate(inputs,config,tokens_to_gen)

    return output_ids

'''
out_toks = main("Alan Turing theorized that computers would one day become",config)
print("Decoded output:",50*'-')
#print(tokenizer.decode(greedy_output[0],skip_special_tokens=True))
print(out_toks)


"""## Data reformatting for Training

### CBT-cn
"""

from datasets import load_dataset

cbt_data_train = load_dataset("cbt", "CN",split="train[:10]")
cbt_data_test = load_dataset("cbt", "CN",split="test")
cbt_data_validate = load_dataset("cbt", "CN",split="validation[:50]")

cbt_raw_train =  load_dataset("cbt", "raw",split="train")

#print(cbt_data['train'])
#print(cbt_data['train'][0]['question']

cbt_data_train = cbt_data_train.add_column("label",[q.index(a)for q,a in zip(cbt_data_train["options"],cbt_data_train["answer"])])
cbt_data_test = cbt_data_test.add_column("label",[q.index(a)for q,a in zip(cbt_data_test["options"],cbt_data_test["answer"])])
cbt_data_validate = cbt_data_validate.add_column("label",[q.index(a)for q,a in zip(cbt_data_validate["options"],cbt_data_validate["answer"])])


def preprocess_function(examples):
    # Construct context strings for each example
    first_sentences = [[''.join(context) + '$'] * len(examples['options'][i]) for i,context in enumerate(examples["sentences"])]
    second_sentences = [
        [ question.replace('XXXXX',option) for option in examples['options'][i]] for i, question in enumerate(examples["question"])
    ]
    labels = [ q.index(a) for q,a in zip(examples["options"],examples["answer"])]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])


    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    return {k: [v[i : i + 10] for i in range(0, len(v), 10)] for k, v in tokenized_examples.items()}



#tokenized_cbt_train = cbt_data_train.map(preprocess_function,batched=True)
#tokenized_cbt_test = cbt_data_test.map(preprocess_function,batched=True)
#tokenized_cbt_validate = cbt_data_validate.map(preprocess_function,batched=True)

cbt_raw_dataset = cbt_raw_train.train_test_split(test_size=0.2)

"""## Wikitext-103"""

from datasets import load_dataset

dataset = load_dataset("wikitext",'wikitext-103-v1',split="train[:2000]")
dataset = dataset.train_test_split(test_size=0.2)
print(dataset["train"][0].keys())

"""### Preprocess by tokenizing and chunking text"""


def tokenize_function(examples):
  outputs = tokenizer(examples["text"],
                      padding="max_length",
                      truncation=True,
                      max_length=1024,
                      return_overflowing_tokens=True,
                      return_length=True)
  input_batch = []
  for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
      if length == 1024:
          input_batch.append(input_ids)
  return {"input_ids": input_batch, "labels":input_batch.copy()}

tokenized_dataset = dataset.map(tokenize_function,
                                batched=True,
                                num_proc=4,
                                remove_columns=dataset['train'].column_names
                                )

def cbt_tokenize_function(examples):
  outputs = tokenizer(examples["content"],
                      padding="max_length",
                      truncation=True,
                      max_length=256,
                      return_overflowing_tokens=True,
                      return_length=True)
  input_batch = []
  for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
    if length == 256:
        input_batch.append(input_ids)
  return {"input_ids": input_batch, "labels":input_batch.copy()}

tokenized_raw_cbt = cbt_raw_dataset.map(cbt_tokenize_function,
                                batched=True,
                                num_proc=4,
                                remove_columns=cbt_raw_dataset['train'].column_names
                                )

print(tokenized_raw_cbt)

from torch.utils.data import DataLoader

if "attention_mask" in tokenized_dataset.keys():
    tokenized_dataset = tokenized_dataset.remove_columns(["attention_mask"])
print(tokenized_dataset["train"][1].keys())
tokenized_dataset.set_format("torch")

# Shuffle data
small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(2000))
small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(400))

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up Dataloaders
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=2,collate_fn=data_collator)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=2,collate_fn=data_collator)

print(len(train_dataloader))

from torch.utils.data import DataLoader

if "attention_mask" in tokenized_raw_cbt.keys():
    tokenized_raw_cbt = tokenized_raw_cbt.remove_columns(["attention_mask"])
print(tokenized_raw_cbt["train"][1].keys())
tokenized_raw_cbt.set_format("torch")

# Shuffle data
small_train_dataset = tokenized_raw_cbt["train"].shuffle(seed=42).select(range(len( tokenized_raw_cbt["train"])))
small_eval_dataset = tokenized_raw_cbt["test"].shuffle(seed=42).select(range(100))

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up Dataloaders
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=2,collate_fn=data_collator)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=2,collate_fn=data_collator)

print(len(train_dataloader))

"""### Collator"""

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        #label_name = "answer"
        batch_size = len(features)

        labels = [feature.pop("label") for feature in features]
        num_choices = len(features[0]["input_ids"])
        #for k,v in features[0].items():
          #print("asd",k,v[0])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        #if len(batch['input_ids']) > 1:
          #print("HERE:",batch_size, len(batch['input_ids'][0]))
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["mc_labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
  preds, labels = eval_pred
  #print(f"EVAL PRED: {type(preds[0])} { type(preds[1])}")
  preds = np.argmax(preds[1],axis=1)
  return accuracy.compute(predictions=preds,references=labels)

"""## Cleanup"""

#del lm_scorer
#del batch
#del outputs
#del perplexity
#del eval_loss
#del res
del loss
del optimizer
del accelerator
del logits
del lr_scheduler
del train_dataloader
del eval_dataloader
del tokenizer
del model
del gpt2

"""### CUDA cache clear"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2
import gc
gc.collect()
torch.cuda.empty_cache()
print('Memory Usage:')
print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
print('max:   ', round(torch.cuda.max_memory_reserved(0)/1024**3,1), 'GB')

"""## Fine-Tuning

## CBT-cn
"""

import evaluate
from transformers import AutoTokenizer, GPT2DoubleHeadsModel,Trainer,TrainingArguments
from transformers import Trainer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = GPT2DoubleHeadsModel.from_pretrained("openai-community/gpt2")

tokenizer.pad_token = tokenizer.eos_token

class MCTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("mc_labels")
        #print("labels",labels)
        outputs = model(**inputs)
        logits = outputs.get('mc_logits')
        loss_fct = torch.nn.CrossEntropyLoss()
        #print("ASDASD")
        loss = loss_fct(logits, labels)
        outputs['logits'] = outputs['mc_logits']
        #print("EXIT", outputs['logits'].shape)
        return (loss,outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="my_awesome_cbt_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    fp16=True,
    label_names=["mc_labels"],
    optim="adafactor"
)

trainer = MCTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_cbt_train,
    eval_dataset=tokenized_cbt_validate,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)
trainer.train()

"""## Wikitext-103

## Optimizer setup
"""

from torch.optim import AdamW
from accelerate import Accelerator

#optimizer = AdamW(model.parameters(), lr=5e-5)



accelerator = Accelerator()

config = Config(
    12,
    768,
    0.1,
    12,
    3072,
    1024,
    len(tokenizer),
    200.,
    100.,
    torch_device
)
weights = { k : v for k,v in model.state_dict().items()}

starting_weight_range_beta = [-90,-110,-10,-100,-30,-140,-137,-130,-155,-150,-150,-150]
starting_weight_level_beta = [0]*12

starting_weight_range_gamma = [10000,10000,1000,10000,1000,1000,100,100,10,20,10,10]

starting_weight_level_beta[1] = -3



for i in range(12):
  weights[f"consmax.{i}.beta"] = [[config.beta_start] for j in range(config.n_heads)]
  weights[f"consmax.{i}.gamma"] = [[config.gamma_start] for _ in range(config.n_heads)]

print(weights[f"consmax.11.beta"])
print(starting_weight_range_beta)
gpt2 = GPT2_LM(config,weights).to(config.device)

optimizer = AdamW(gpt2.parameters(), lr=5e-5)

gpt2, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    gpt2, optimizer, train_dataloader, eval_dataloader
)

#accelerator.load_state()

# model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
#     model, optimizer, train_dataloader, eval_dataloader
# )

from transformers import get_scheduler

num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1_00,
    num_training_steps=num_training_steps,
)

"""## Final model adjustments"""

def evaluate(model):
    model.eval()
    losses = []
    for step, batch in tqdm(enumerate(eval_dataloader),total=len(eval_dataloader)):
      with torch.no_grad():
          outputs = model(batch["input_ids"].to(device), labels=batch["input_ids"].to(device))
      print(outputs.loss)
      losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.stack(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()
#model.to('cuda')
#res = evaluate(gpt2)
#print(res)

from tqdm.auto import tqdm

gpt2.train()
model.train()

def calc_loss(inputs, logits,alpha=1.0):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


gradient_accumulation_steps = 8
eval_steps = 5_000

#torch.autograd.set_detect_anomaly(True)
gpt2.train()
#model.train()
completed_steps = 0
total_loss = 0
for epoch in range(num_train_epochs):
    for step, batch in tqdm(
        enumerate(train_dataloader, start=1), total=num_training_steps
    ):
        outputs = gpt2(batch["input_ids"],labels=batch["input_ids"])
        loss,logits = outputs.loss,outputs.logits
        if step % 10 == 0:
            accelerator.print(
                {
                    "samples": step * 2,
                    "steps": completed_steps,
                    "loss/train": total_loss / step,
                }
            )
        print(loss)
        if loss.isnan().any():
          del loss
          lr_scheduler.step()
          optimizer.zero_grad()
          continue
        total_loss += loss.item()
        accelerator.backward(loss)
        for param in gpt2.parameters():
          if (step % 50) == 0 and param.shape == (12,1):
            print(param,param.grad)
        if optimizer:
            accelerator.clip_grad_norm_(gpt2.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
        if (step % 500) == 0:
            eval_loss, perplexity = evaluate(gpt2)
            accelerator.save_state("my_awesome_gpt2_model/")
            print({"loss/eval": eval_loss, "perplexity": perplexity})
            run_cbt_bench()
            gpt2.train()
            accelerator.wait_for_everyone()

from huggingface_hub import HfApi,login
api = HfApi()

login('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')

api.upload_folder(
                    folder_path="my_awesome_gpt2_model/",
                    path_in_repo=f"colab_testing/",
                    repo_id="tmleong/gpt2_consmax_approx",
                )

"""## Accuracy evaluation"""

for param in gpt2.parameters():
  if param.shape == (12,1):
    print(param,param.shape)

"""## CBT-cn Evaluation

### Load model from hub
"""

!pip install safetensors

from huggingface_hub import hf_hub_download,login
login('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')


path = hf_hub_download(repo_id="tmleong/gpt2_consmax_approx", filename="epoch_0_checkpoint_38800/model.safetensors")

print(path)

from safetensors.torch import load_model, save_model, safe_open

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
    12,
    768,
    0.1,
    12,
    3072,
    1024,
    len(tokenizer),
    55.,
    100.,
    torch_device
)

weights = { k : v for k,v in model.state_dict().items()}

for i in range(12):
  weights[f"consmax.{i}.beta"] = [[config.beta_start] for j in range(config.n_heads)]
  weights[f"consmax.{i}.gamma"] = [[config.gamma_start] for _ in range(config.n_heads)]

gpt2 = GPT2_LM(config,weights)


for k,v in gpt2.state_dict().items():
  if 'smax' in k:
    print(v)

"""### Benchmark setup"""

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from tqdm import tqdm
class GPT2LMScorer():
    def _build(self, model,tokenizer,batch_size=10,device='cpu') -> None:
        if batch_size < 1:
            raise ValueError("The batch_size option must be positive")
        # pylint: disable=attribute-defined-outside-init
        self.batch_size = batch_size
        if model is None or tokenizer is None:
          # pylint: disable=attribute-defined-outside-init
          self.tokenizer = AutoTokenizer.from_pretrained(
              "gpt2", use_fast=True, add_special_tokens=False
          )
          # Add the pad token to GPT2 dictionary.
          # len(tokenizer) = vocab_size + 1
          self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
          self.tokenizer.pad_token = "<|pad|>"

          self.model = GPT2LMHeadModel.from_pretrained("gpt2")
          # We need to resize the embedding layer because we added the pad token.
          self.model.resize_token_embeddings(len(self.tokenizer))
        else:
          print('here')
          self.model = model
          self.tokenizer = tokenizer

        self.model.eval()
        self.model.to('cuda')
        self.device=device

    def _add_special_tokens(self, text: str) -> str:
        return self.tokenizer.bos_token + text + self.tokenizer.eos_token

    def _tokens_log_prob_for_batch(
        self, text
    ):
        outputs = []
        if len(text) == 0:
            return outputs

        # TODO: Handle overflowing elements for long sentences
        text = list(map(self._add_special_tokens, text))
        encoding = self.tokenizer.batch_encode_plus(
            text,padding='max_length', return_tensors="pt",
        )
        with torch.no_grad():
            ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
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

            sent_log_probs = sent_log_probs
            sent_ids =  sent_ids
            #print(sent_log_probs)
            output = (sent_log_probs, sent_ids, sent_tokens)
            outputs.append(output)

        return outputs
    def _tokens_log_prob(
        self, text
    ):
        outputs = []
        for i in range(0, len(text), self.batch_size):
            batch = text[i : i + self.batch_size]
            outputs.extend(self._tokens_log_prob_for_batch(batch))
        return outputs

    def sentence_score(
        self, text, log: bool = True, reduce: str = "prod",
    ):
        sentences = [text] if isinstance(text, str) else text
        scores= []
        if len(sentences) == 0:
            return scores

        outputs = self._tokens_log_prob(sentences)
        for output in outputs:
            log_probs = output[0]
            tlen = log_probs.shape[0]

            if reduce == "prod":
                score = log_probs.sum()
            elif reduce == "mean":
                score = log_probs.logsumexp(0) - math.log(tlen)
            elif reduce == "gmean":
                score = log_probs.mean(0)
            elif reduce == "hmean":
                score = log_probs.neg().logsumexp(0).neg() + math.log(tlen)
            else:
                raise ValueError("Unrecognized scoring strategy: %s" % reduce)
            if not log:
                score = score.exp()
            #print(score)
            scores.append(score.item())

        return scores[0] if isinstance(text, str) else scores

#weights = { k : v for k,v in model.state_dict().items()}

#gpt2 = GPT2_LM(config,weights).to(config.device)
lm_scorer = GPT2LMScorer()
lm_scorer._build(gpt2,tokenizer,10,'cuda')

score = lm_scorer.sentence_score(['good luck the bear recipe dog keyboard'])
print(score)

print(len(cbt_data_validate))
print(cbt_data_validate[0]['sentences'])

"""### Evaluation"""

def run_cbt_bench():
  lm_scorer = GPT2LMScorer()
  gpt2.eval()
  lm_scorer._build(gpt2,tokenizer,10,'cuda')

  correct = 0
  print(range(0,len(cbt_data_validate),5))
  for i in tqdm(range(0,len(cbt_data_validate),5)):
      rows = cbt_data_validate[i:i+5]

      formatted_text = [[(''.join(rows['sentences'][j]) + rows['question'][j].replace('XXXXX',option)).replace(' .','.') for option in rows['options'][j]] for j in range(5)]
      flattened = sum(formatted_text, [])
      #print("Len falttened",len(flattened))
      scores = lm_scorer.sentence_score(flattened)
      #print("Len scores",len(flattened))
      for j in range(5):
          #print(actual,len(scores),j*10,(j+1)*10)
          scores_tmp = scores[j*10:(j+1)*10]
          actual = rows['options'][j].index(rows['answer'][j])
          pred = torch.argmax(torch.tensor(scores_tmp).cpu().detach()).item()
          #print(scores,pred,actual)
          correct += (actual == pred)
      if i > 0:
          print("\n RESULTS: \n",correct,i,correct / i)

  print("\n RESULTS: \n",correct,len(cbt_data_validate),correct / len(cbt_data_validate))
  gpt2.train()

#run_cbt_bench()
'''