import numpy as np
import attn
import layers
import pack
import torch

import gpt2_approx_checkpoint_2 as gpt2_ref

# Preprocessing required to transform raw input data into form that can be used as input

# Input:  Text input, model weight dict 
# Output: Embedded input, Weight dictionary
def gpt2_setup(tokens,config,weights):

    # Directly embed using embedding layer
    embedded= None
    embedding_layer = gpt2_ref.EmbeddingLayer(config,weights)
    with torch.no_grad():
        embedded = embedding_layer(tokens)
    
    w_out,b_out = torch.randn((768,768)), torch.randn((768,)) / 1000

    

    new_weights = {}
    # Convert Weights into the appropriate format for each layer
    for k,val in weights.items():
        v = val.cpu().detach().numpy()
        #Layer norm case Should be (768,) in either case
        if "ln_"in k:
            new_weights[k] = pack.expand_bias(weights[k])
        elif "c_attn.weight" in k:
            qw, kw, vw = torch.split(val, 768, dim=-1)
            qw = qw.cpu().detach().numpy()
            kw = kw.cpu().detach().numpy()
            vw = vw.cpu().detach().numpy()

            new_weights[k+"_q"] = pack.pack_from_row(qw.T)
            new_weights[k+"_k"] = pack.pack_from_row(kw.T)
            new_weights[k+"_v"] = pack.pack_from_row(vw.T)
        
        elif "c_attn.bias" in k: 
            qb, kb, vb = torch.split(val, 768, dim=-1)

            qb = qb.cpu().detach().numpy()
            kb = kb.cpu().detach().numpy()
            vb = vb.cpu().detach().numpy()

            new_weights[k+"_q"] = pack.expand_bias_head_row(qb.T,config.n_heads)
            new_weights[k+"_k"] = pack.expand_bias_head_row(kb.T,config.n_heads)
            new_weights[k+"_v"] = pack.expand_bias_head_col(vb.T,config.n_heads,128,64)
        elif "c_proj.weight" in k:
            new_weights[k] = pack.pack_from_row(v.T)
        elif "c_proj.bias" in k:
            new_weights[k] = pack.expand_bias(v)
        elif "mlp.c_fc.weight" in k:
            new_weights[k] = pack.pack_from_row(v.T)
        elif "mlp.c_fc.bias" in  k:
            new_weights[k] = pack.expand_bias(v)
        elif "mlp.c_proj.weight" in k:
            new_weights[k] =  pack.pack_from_row(v.T)
        elif "mlp.c_proj.bias" in k:
            new_weights[k] = pack.expand_bias(v)
    
    return embedded[0].cpu().detach().numpy(),new_weights
            
# End-to-End GPT-2 Inference
# Assumes Pre-embedded input
# Outputs numpy version of output 
def gpt2_inference(embed_in,config,weights):

    assert embed_in.shape == (128,768)
    layer_in = pack.pack_from_row(embed_in)
    
    print("\nBeginning GPT2-CKKS inference","-"*50,"\n")
    mask = torch.tril(torch.ones((config.max_len, config.max_len)))
    mask2 = torch.triu(torch.ones((config.max_len, config.max_len)),diagonal=1)
    
    mask_packed = pack.pack_from_row(mask2)
    ex_mask_packed = pack.pack_from_row(mask)
    for idx in range(config.n_layers):

        print(f"LAYER IN: {layer_in[0][:10]}")
        # Layer Norm
        ln1_res = layers.layer_norm(layer_in,
                                    weights[f"transformer.h.{idx}.ln_1.weight"],
                                    weights[f"transformer.h.{idx}.ln_1.bias"],
                                    768,40298902)

        print(f"RES LN1 OUT: {ln1_res[0][:10]}")
        # Attn Layer
        attn_res = attn.attention_layer(ln1_res,
            weights[f"transformer.h.{idx}.attn.c_attn.weight_q"],
            weights[f"transformer.h.{idx}.attn.c_attn.bias_q"],
            weights[f"transformer.h.{idx}.attn.c_attn.weight_k"],
            weights[f"transformer.h.{idx}.attn.c_attn.bias_k"],
            weights[f"transformer.h.{idx}.attn.c_attn.weight_v"],
            weights[f"transformer.h.{idx}.attn.c_attn.bias_v"],
            weights[f"transformer.h.{idx}.attn.c_proj.weight"],
            weights[f"transformer.h.{idx}.attn.c_proj.bias"],
            mask_packed,ex_mask_packed
        )

        print(f"RES ATTN OUT: {attn_res[0][:10]}")

        print(f"LAYER IN: {layer_in[0][:10]}")
        # Residual Layer
        for i in range(len(ln1_res)):
            attn_res[i] += layer_in[i]
        
        print(f"RES RESID OUT: {attn_res[0][:10]}")
        
        # Second LayerNorm
        ln2_res = layers.layer_norm(attn_res,
                                    weights[f"transformer.h.{idx}.ln_2.weight"],
                                    weights[f"transformer.h.{idx}.ln_2.bias"],
                                    768,5.74156342e+08)

        print(f"RES LN 2 OUT: {ln2_res[0][:10]}")

        # MLP
        mlp_res = layers.mlp(ln2_res,
                             weights[f"transformer.h.{idx}.mlp.c_fc.weight"],
                             weights[f"transformer.h.{idx}.mlp.c_fc.bias"],
                             weights[f"transformer.h.{idx}.mlp.c_proj.weight"],
                             weights[f"transformer.h.{idx}.mlp.c_proj.bias"])
        print(f"RES MLP OUT: {mlp_res[0][:10]}")
        print(f"ATTN MLP OUT: {attn_res[0][:10]}")

        # Residual layer
        for i in range(len(mlp_res)):
            layer_in[i] = mlp_res[i] + attn_res[i]

        print(f"RES RESID 2 OUT: {layer_in[0][:10]}")

    # Final Layer norm
    out = layers.layer_norm(layer_in,
                                    weights["transformer.ln_f.weight"],
                                    weights["transformer.ln_f.bias"],
                                    768,1.1047115e+10)
    return out

