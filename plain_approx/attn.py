import copy
import numpy as np
import fold
import numpy.random as rand
import math
import iterations as iter
import matrix_mul as matmul
import torch
import torch.nn as nn
import pack

def mask_out(arr,bit_range):
    #bit_range is specified as (start,len)
    mask = np.zeros_like(arr,dtype=float)
    for i in range(bit_range[0],bit_range[0]+bit_range[1]):
        mask[i] = 1.0
    out = arr * mask
    return out

def round_to_2(x):
    return math.pow(2,math.ceil(math.log(x,2)))

def attn_convert_to_prefold(arr,seq_len,hidden_dim):
    h = int(round_to_2(hidden_dim))
    x = np.zeros((h*2*seq_len // 32768,32768))
    # Can pack to 42 ciphertexts into first 3
    global_idx = 0
    offsets = [(0, 32768 % hidden_dim),(hidden_dim - (32768 % hidden_dim),hidden_dim - (32768 % hidden_dim)),(32768 % hidden_dim,0)] # (begin_rot_amount,amount hanging off end)
    for i in range(3):
        arr[i] = np.roll(arr[i],-offsets[i][0])
        for k in range(42):
            # Mask out row
            masked_out = mask_out(arr[i],(0,hidden_dim))
            # Place row into row-packed cipher
            #print(f"x idx: {k} {global_idx // 32768} {global_idx % 32768} {masked_out[:10]}")
            x[global_idx // 32768] += np.roll(masked_out,global_idx % 32768)

            # Update global index
            global_idx += h*2
            arr[i] = np.roll(arr[i],-hidden_dim)
        
        # Hanging case
        if i == 2:
            break
        #print("global_idx0 ",global_idx,"---\n\n")
        masked_out = mask_out(arr[i],(0,offsets[i][1]))
        x[global_idx // 32768] += np.roll(masked_out,global_idx % 32768)
        global_idx += offsets[i][1]
        
        # Pack remaining part of matrix into remaining part of ciphertext
        if i != len(offsets)-1:
            masked_out = mask_out(arr[i+1],(0,offsets[i+1][0]))
            x[global_idx // 32768] += np.roll(masked_out,global_idx % 32768)
            global_idx += offsets[i+1][0] + (2*h-hidden_dim)
        arr[i] = np.roll(arr[i],-offsets[i][1])
    return x


# Pack weight columns into 24 ciphers
def pack_weights(weights):
    assert weights.shape == (768,768)
    x = np.zeros((48,32768))

    global_idx = 0
    
    for i in range(768):
        tmp = np.zeros((768))
        tmp+=weights[i]
        tmp.resize((32768))
        x[global_idx // 32768] += np.roll(tmp,global_idx % 32768)
        global_idx += 2048

    return x   


# Very expensive matmul occurs here
def expensive_matrix_mul_row(prefold_ciphers,weights,bias):
    assert prefold_ciphers.shape == (8,32768)
    assert weights.shape == (48,32768)

    # Output is (12,128,64) matrix Q,K
    output = np.zeros((12,32768))
    for i in range(8):
        for j in range(48):
            for rots in range(16):
                rolled = np.roll(weights[j],-(rots*2048))
                res1 = prefold_ciphers[i] * rolled
                #print(prefold_ciphers[i][758:768],rolled[758:768],res[758:768])
                # Format for fold
                #print("res1", (res1[1024:2048]==0).all())
                res = res1 + np.roll(res1,1024)
                #print("res", (res[:1024]==res[1024:2048]).all())
                folded = fold.quickSum(res,2048)
                
                for pos in range(16):
                    row = i * 16 + pos
                    col = j * 16 + ((rots + pos) % 16)

                    abs_pos = row*768 + col

                    head_row = abs_pos // 64
                    head_col = abs_pos % 64
                    head = head_row % 12

                    masked_out = mask_out(folded,(pos*2048,1))
                    
                    desired_location = row*128 + head_col
                    
                    shift_amt = desired_location - pos*2048
                    rolled = np.roll(masked_out, shift_amt)
                    # if head == 0 and rolled[128] != 0:
                    #     print("\nAdjusting head 0")
                    #     print(i,j,"row ",row,col,"rots ",rots,pos,"hrow ",head_row,head_col,abs_pos,shift_amt, desired_location, rolled[128])
                        #print(np.roll(mask_out(res1,-(pos*2048))))
                    
                    #print(i,j,rots,pos,head_row,head_col, abs_pos,desired_location)

                    output[head] += rolled
    for i in range(output.shape[0]):
        output[i] += bias[i]
    #print(f"bias: {bias}\n")

    return output

# Very expensive matmul occurs here
def expensive_matrix_mul_col(prefold_ciphers,weights,bias):
    assert prefold_ciphers.shape == (8,32768)
    assert weights.shape == (48,32768)

    # Output is (12,128,64) matrix V column packed
    output = np.zeros((12,32768))
    for i in range(8):
        for j in range(48):
            for rots in range(16):
                rolled = np.roll(weights[j],-(rots*2048))
                res1 = prefold_ciphers[i] * rolled
                #print(prefold_ciphers[i][758:768],rolled[758:768],res[758:768])
                # Format for fold
                res = res1 + np.roll(res1,1024)
                folded = fold.quickSum(res,2048)
                
                for pos in range(16):
                    row = i * 16 + pos
                    col = j * 16 + ((rots + pos) % 16)

                    abs_pos = row*768 + col

                    head_row = abs_pos // 64
                    head_col = abs_pos % 64
                    head = head_row % 12

                    masked_out = mask_out(folded,(pos*2048,1))
                    
                    desired_location = head_col*256 + row
                    
                    shift_amt = desired_location - pos*2048
                    rolled = np.roll(masked_out, shift_amt)
                    
                    #print(i,j,rots,pos,head_row,head_col, abs_pos,desired_location)

                    output[head] += rolled
    for i in range(output.shape[0]):
        output[i] += bias[i]
    #print(f"bias: {bias}\n")
    
    return output

def qk_matmul(Q,K):
    assert Q.shape == (12,32768)
    assert K.shape == (12,32768)
    
    # Output is (12,128,128) matrices that result from Q^KT
    output = np.zeros((12,32768))
    for i in range(12):
        # Need to create duplicate format for matmul to work
        # |A|B|C|...|A|B|C|
        dup = K[i] + np.roll(K[i],16384)
        assert np.equal(dup[:16384],dup[16384:]).all()

        for rots in range(128):
            rolled = np.roll(dup,-(rots*128))
            res1 = Q[i] * rolled
            
            # Format for fold
            res = res1 + np.roll(res1,64)
            folded = fold.quickSum(res,128)
            for pos in range(128):
                row = i * 128 + pos
                col = i * 128 + ((rots + pos) % 128)

                abs_pos = row*128 + col
                head_col = abs_pos % 128

                masked_out = mask_out(folded,(pos*128,1))
                
                desired_location = row*256 + head_col
                
                shift_amt = desired_location - pos*128
                rolled = np.roll(masked_out, shift_amt)
                
                #print(i,j,rots,pos,head_row,head_col, abs_pos,desired_location)

                output[i] += rolled
    return output

def attn_softmax(inputs,attn_mask,extract_mask):

    in_shape = inputs.shape

    assert in_shape == (12,32768)

    outputs = np.zeros(in_shape)
    maxes = np.zeros(in_shape)
    folded = np.zeros(in_shape)

    ones_mask = np.zeros(in_shape[1])
    zeros_mask = np.ones(in_shape[1])
    

    # Tradeoff: subtract out 1s instead of mul by 0
    for i in range(128):
        for j in range(128):
            ones_mask[i*256+128+j] = 1.0
            zeros_mask[i*256+128+j] = 0.0
    
    print(f"Input: {inputs[0][:10]}\n")
    # Compute max and re-zero
    for i in range(in_shape[0]):
        maxes[i] = inputs[i] + np.roll(inputs[i],128)
        #print(f"Duplicate correctness {(maxes[i][:128] == maxes[i][128:256]).all()} {maxes[i][:128].max()}")
        # Normalize to 0
        maxes[i] = fold.quickMax(maxes[i],256)
        maxes[i] = maxes[i] * zeros_mask
    print(f"Maxes: {maxes[0][0:10]} | True max: {inputs[0][0:128].max()}\n")

    # Subtract out max to ensure no blowup
    for i in range(in_shape[0]):
        outputs[i] = inputs[i] - maxes[i]

    print(f"x-max(x): {outputs[0][:10]}\n")

    # Compute exp and subtract out 1s
    for i in range(in_shape[0]):
        exp_res = iter.exp(outputs[i],13)
        outputs[i] = exp_res - ones_mask
        outputs[i] = np.nan_to_num(outputs[i],nan=0.0,posinf=0.0,neginf=0.0)
        outputs[i] *= extract_mask
        
    
    print(f"exp: {outputs[0][:10]} {(outputs[0][128:256] == 0).all()} \n")
    # Compute sum
    for i in range(in_shape[0]):
        folded[i] = outputs[i] + np.roll(outputs[i],128)
        folded[i] = fold.quickSum(folded[i],256)
    print(f"folded: {folded[0][:10]} {outputs[0][:128].sum()} {outputs[0][:128].sum()}\n")
    
    tmp_val = inputs[0][:64]
    tmp_max = tmp_val.max()
    tmp_res = np.exp(tmp_val-tmp_max)/np.exp(tmp_val-tmp_max).sum()
    print("Expected fold: ",tmp_res[:10])
    # Compute quotient
    for i in range(in_shape[0]):
        div_term = pow(2,20)
        numerator,denominator = outputs[i]/div_term,folded[i]/div_term
        outputs[i] = iter.goldschmidt_division(numerator,denominator,13)
        outputs[i] = np.nan_to_num(outputs[i],nan=0.0,posinf=0.0,neginf=0.0)
        outputs[i] = outputs[i] * zeros_mask
    return outputs

# Output appended packing
def sv_matmul(S,V):
    print(S.shape)
    assert S.shape == (12,32768) 
    assert V.shape == (12,32768)

    # Output is (8,32768) matrix
    output = np.zeros((8,32768))
    for i in range(12):
        for rots in range(64):

            # Need to create duplicate format for rotates to work
            # |A|B|C|...|A|B|C|
            dup = V[i] + np.roll(V[i],16384)

            rolled = np.roll(dup,-(rots*256))
            res1 = S[i] * rolled

            # Format for fold
            res = res1 + np.roll(res1,128)
            folded = fold.quickSum(res,256)

            for pos in range(128):
                row = pos
                col = ((rots + pos) % 64)

                
                # 1. Compute which chunk of size 768 to insert
                chunk_pos = row
                # 2. Compute where in chunk to place element
                chunk_offset = (i * 64) + col
                #print(i,rots,pos,chunk_pos)
                # 3. Compute which ciphertext the element belongs in
                cipher_idx = chunk_pos // 16
                # 4. Compute desired_location within cipher
                desired_location = (chunk_pos % 16)*2048 + chunk_offset
                

                #print(i,j,rots,pos,head_row,head_col, abs_pos,desired_location)
                masked_out = mask_out(folded,(pos*256,1))
                
                
                shift_amt = desired_location - pos*256
                rolled = np.roll(masked_out, shift_amt)

                output[cipher_idx] += rolled
    return output




# Inputs are assumed to be row-packed.
# Weights are assumed to be column-packed.
# Mask to ensure future tokens can't attend to those in the past
def attention_layer(inputs,qw,qb,kw,kb,vw,vb,w_out,b_out,mask,ex_mask):
    """
    Performs Attention Layer
    
    Inputs:
    inputs: embedded vectors (batch,seq_len,hidden) packed into 8 ctxts
    qw,kw,vw: weight vectors of Query, Key and Value matrices
    qb,kb,vb: bias vectors of Query, Key and Value matrices
    """

    # Assume fold packing
    assert inputs.shape == (8,32768)

    # Project onto attention layer heads
    Q = expensive_matrix_mul_row(inputs,qw,qb)
    K = expensive_matrix_mul_row(inputs,kw,kb)
    V = expensive_matrix_mul_col(inputs,vw,vb)

    print(f"RES Q: {Q[0][128:138]}\n")
    print(f"RES K: {K[0][128:138]}\n")
    #return np.zeros((8,32768))

    Q_tmp = pack.unpack_heads(Q,12,128,64)
    K_tmp = pack.unpack_heads(K,12,128,64)

    #print(f"QK unpacked RES : {(Q_tmp @ np.transpose(K_tmp,axes=(0,2,1)))[0][0][:10]}")

    # Perform QK^T multiplications
    #Input: (12,128,64) x (12,128,64)
    #Output: (12,128,128)
    QKt = qk_matmul(Q,K)
    
    print(f"QK MATMUL RES: {QKt[0][:10]}")

    # Divide by sqrt(head dim)
    QKt = QKt / (math.sqrt(64))


    #print(f"QKT shape: {QKt.shape} mask shape: {mask.shape} {mask[0][:10]}")
    # Add Mask
    for i in range(len(QKt)):
        QKt[i] += mask[0] * -1e5

    #print(f"QKT shape: {QKt[0][:10]} mask shape: {mask.shape}")
    # Softmax
    # Expects Row-wise packing
    # Input: (12,128,128)
    # Output: (12,128,128)
    #softmax_out = attn_softmax(QKt,mask[0],ex_mask[0])
    QKt = pack.unpack_heads(QKt,12,128,128)
    #print(f"RES PRE-SMAX SCORES: {QKt[0][0][:10]}")
    softmax_out = torch.softmax(torch.tensor(QKt),dim=-1).detach().numpy()
    softmax_out = pack.pack_heads(softmax_out,12,128,128)
    # Multiply by V and concat
    pre_out = sv_matmul(softmax_out,V)
    
    # Output
    return matmul.generic_matrix_mul(pre_out,w_out,b_out,128,768,768,768)


def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value,smax,mask=None, dropout=None):
    'compute scaled dot product attention'
    d_k = query.size(-1)

    #print(f"QUERY/KEY EXP: {query[0][0][:10]} \n{key[0][0][:10]}\n")

    qk = torch.matmul(query,key.transpose(1,2))
    assert (qk == (query @ key.transpose(1,2))).all()
    print(f"\nQK EXP: {qk[0][0][:10]}\n")
    
    scores = qk / math.sqrt(d_k) # QK^T/sqrt(d_k). Note since first dim is batch, we need to transpose last two
    #print(f"score_dim:{scores.shape} {query.shape} mask_dim:{mask.shape}")
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e10)

    #print(f"score_dim_after_fill:{scores.shape}")
    #print(f"EXP PRE-SMAX SCORES: {scores[0][0][:10]}")
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
    def __init__(self,config,weights):
        super(MultiHeadAttention,self).__init__()

        self.d_k = config.d_model // config.n_heads # dimension for each head
        self.heads = config.n_heads # number of attention heads

        self.linears = clones(torch.nn.Linear(config.d_model,config.d_model),4)
        z = weights["qb"]
        print(f"bias1: {z[:10]}\n")
        with torch.no_grad():
          self.linears[0].weight = nn.Parameter(weights["qw"].T) # W_q
          self.linears[0].bias = nn.Parameter(weights["qb"]) # q_bias

          self.linears[1].weight = nn.Parameter(weights["kw"].T) # W_k
          self.linears[1].bias = nn.Parameter(weights["kb"]) # k_bias

          self.linears[2].weight = nn.Parameter(weights["vw"].T) # W_v
          self.linears[2].bias = nn.Parameter(weights["vb"])

          self.linears[3].weight = nn.Parameter(weights[f"c_proj.weight"].T)
          self.linears[3].bias = nn.Parameter(weights[f"c_proj.bias"])

    def forward(self,query,key,value,mask=None):

        #print(f"q:{query.shape}key:{key.shape} val:{value.shape}\n")

        #tmp =  key @ self.linears[0].weight +  self.linears[0].bias
        #print(f"KW: {tmp[0][0][:10]}")

        query,key,value = \
        [l(x).view(128,12,64).transpose(0,1) # Ensure last two dims of Q,K,V are T x d/k
            for l, x in zip(self.linears, (query,key,value))]

        print(f"EXP WK: {key[0][1][:10]}\n")
        print(f"EXP WQ: {query[0][1][:10]}\n")
        
        #print(f"WV: {value[0][0][0][:10]}")

        attn,_ = attention(query,key,value,None,mask,None)
        #qk = attention(query,key,value,None,mask,None)
        #print(f"attn out: {attn[0][0][0][:10]}")

        #print(f"softmax output: {attn[0][0][0][:20]}")

        # Concat output for feedforward layer
        x = attn.transpose(0,1).contiguous().view(128,self.heads * self.d_k)
        #print(f"grouped output attention: {x[0][0][:20]}")

        x = self.linears[3](x)
        #print(f"Attn out: {x[0][0][:10]}")
        return x


if __name__ == "__main__":
    print("Running attn")