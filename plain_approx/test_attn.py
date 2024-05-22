import pytest
import numpy as np
import matrix_mul as matmul
import iterations as iter
import attn
import numpy.random as rand
import fold
import torch
import pack

@pytest.mark.sqrt
def test_sqrt():
    num = 25
    assert np.sqrt(num) == 5

@pytest.mark.mul
def test_mul():
    a,b = 4,6
    assert(a*b == 24)

@pytest.mark.matrix
def test_row_matrix_mul():
    input_matrix = np.array([[1,2,3],[4,5,6]])
    weights = np.array([[1,1,1],[2,2,2],[3,3,3]])

    expect = input_matrix @ weights
    res = matmul.row_matrix_mul(input_matrix,weights)
    assert (expect == res).all()

#@pytest.mark.matrix
def test_col_matrix_mul():
    left_matrix = np.array([[1,2,3,0],[4,5,6,0],[7,8,9,0],[10,11,12,0]])
    right_matrix = np.array([[1,1,1,0],[2,2,2,0],[3,3,3,0],[4,4,4,0]])

    expect = left_matrix @ right_matrix.T

    res1 = matmul.col_matrix_mul(left_matrix,right_matrix)
    assert len(res1) == 4
    res = matmul.diagonal_to_row(res1)
    print(res,expect)
    assert (expect == res).all()

#@pytest.mark.attn
def test_pack_weights():
    vec = np.arange(0,768*768).reshape(768, 768)
    vec = attn.pack_weights(vec)
    assert (vec[0][:10] == np.arange(0,10)).all()
    assert (vec[0][768:2048] == np.zeros((2048-768))).all()

#@pytest.mark.attn
def test_expensive_matmul_row():
    vec1 = rand.random((768,768))
    weights = attn.pack_weights(vec1.T)

    vec = rand.rand(3,32768)
    vec2 = vec
    res = attn.attn_convert_to_prefold(vec,128,768)

    result = attn.expensive_matrix_mul_row(res,weights,np.zeros((32768,)))
    assert result.shape == (12,32768)

    #print("Result ", result[1][128:138])

    exp = torch.tensor(vec.reshape((128,768))) @ torch.tensor(vec1)
    exp = exp.view((128,12,64)).transpose(0,1).detach().numpy()

    res_unpacked = pack.unpack_heads(result,12,128,64)
    print(f"RES: {res_unpacked[0][0][:10]}")
    print(f"EXP: {exp[0][0][:10]}")
    assert np.isclose(res_unpacked,exp).all()

    res2 = attn.qk_matmul(result,result)

    res2_unpacked = pack.unpack_heads(res2,12,128,128)

    exp2 = exp @ np.transpose(exp,axes=(0,2,1))

    assert np.isclose(res2_unpacked,exp2).all()
    

#@pytest.mark.attn
def test_expensive_matmul_col():
    vec1 = rand.random((768,768))
    weights = attn.pack_weights(vec1.T)

    vec = rand.rand(3,32768)
    vec[0] = np.arange(0,32768)
    vec[1] = np.arange(0,32768)
    vec[2] = np.arange(0,32768)
    vec2 = vec
    res = attn.attn_convert_to_prefold(vec2,128,768)

    result = attn.expensive_matrix_mul_col(res,weights)
    assert result.shape == (12,32768)

    print("Result ", result[3][256:266])
    exp = vec.reshape((128,768)) @ vec1
    exp = np.transpose(exp.reshape((128,12,64)),axes=(1,0,2)).transpose(0,2,1)
    print("Exp: ", exp[3][1][:10], exp.shape)

def pack_heads(A,num_ciphers,num_rows,row_size):
    out = np.zeros((num_ciphers,32768))
    for i in range(num_ciphers):
        for j in range(num_rows):
            padded = np.pad(A[i][j], (0,32768-row_size),'constant')
            out[i] += np.roll(padded, (row_size*2)*j)
    return out

#@pytest.mark.attn
def test_qk_matmul():
    Q = rand.random((12,128,64))
    K = rand.random((12,128,64))

    Q_packed = pack_heads(Q,12,128,64)
    K_packed = pack_heads(K,12,128,64)

    res = attn.qk_matmul(Q_packed,K_packed)
    exp = Q @ K.transpose(0,2,1)
   
    print(f"RES: {res[1][256:266]}")
    print(f"EXP: {exp[1][1][:10]} {exp.shape}")

    print(res.shape,exp.shape)
    res = pack.unpack_heads(res,12,128,128)
    

    assert np.isclose(res,exp,atol=0.000001).all()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    maxi = np.max(x,keepdims=True,axis=-1)
    return np.exp(x-maxi) / np.sum(np.exp(x-maxi),keepdims=True, axis=-1)

#@pytest.mark.attn
def test_attn_softmax():
    input = rand.random((12,128,128))
    packed = pack_heads(input,12,128,128)

    print((packed[0][0:128] == input[0][0]).all())
    res = attn.attn_softmax(packed)
    exp = softmax(input)
    print(f"RES: {res[0][:10]}\n")
    print(f"EXP: {exp[0][0][:10]}\n")

def compress_for_check(A):
    assert A.shape == (8,32768)
    out = np.zeros((128*768))
    for i in range(8):
        for j in range(16):
            masked_out = attn.mask_out(A[i],(j*2048,768))
            masked_out = np.pad(masked_out,(0,98304-32768),'constant')
            rolled = np.roll(masked_out,(i*12288+j*768) - j*2048)
            out += rolled
    return out
#@pytest.mark.attn
def test_sv_matmul():
    # In: (12,128,64) x (12,128,64)
    # Out: (128,768)
    S = rand.random((12,128,128))
    V = rand.random((12,128,64))

    S_packed = pack_heads(S,12,128,128)

    # Need to column-pack for matrix-mul
    V_packed = pack_heads(V.transpose(0,2,1),12,64,128)

    res = attn.sv_matmul(S_packed,V_packed)
    assert res.shape == (8,32768)

    exp = (S @ V).transpose(1,0,2).reshape(128,768)
    print((S@V).transpose(1,0,2).shape)

    print(f"RES: {res[7][32768-1024-256-10:32768-1024-256]}\n")
    print(f"EXP: {exp[127][758:]}\n")
    print((np.isclose(res[0][:10],exp[0][:10])).all())
    compressed = compress_for_check(res)
    assert np.isclose(compressed,exp.reshape(-1)).all()

#@pytest.mark.attn
def test_out_layer():
    # In: (128,768) x (768,768)
    # Out: (128,768) 
    A = rand.random((128,768))
    W = rand.random((768,768))

    A_packed = pack.pack_from_row(A)
    W_packed = pack.pack_from_row(W.T)

    res = matmul.generic_matrix_mul(A_packed,W_packed,128,768,768,768)
    exp = A @ W

    res = pack.pack_tight(res)

    #print(f"RES: {res[0][2048:2058]}")
    #print(f"EXP: {exp[1][:10]}")
    #assert np.isclose(exp[1][:10],res[0][2048:2058]).all()
    assert np.isclose(exp.reshape(-1),res.reshape(-1)).all()

class Config:
    def __init__(self,seq_len,n_heads,d_model,d_hidden):
      self.seq_len = seq_len
      self.n_heads = n_heads
      self.d_model = d_model
      self.d_hidden = d_hidden

@pytest.mark.attn
def test_attn_layer():
    # In: (128,768)
    # Out: (128,768)
    torch.set_default_dtype(torch.float64)

    config = Config(128,12,768,3072)

    weight_chunk = torch.randn((768,2304)) / 1000
    bias_chunk = torch.randn((2304,)) / 1000

    w_out,b_out = torch.randn((768,768)), torch.randn((768,)) / 1000

    qw, kw, vw = torch.split(weight_chunk, 768, dim=-1)

    qb, kb, vb = torch.split(bias_chunk, 768, dim=-1)

    weights = {
        "qw": qw,
        "qb": qb,
        "kw": kw,
        "kb": kb,
        "vw": vw,
        "vb": vb,
        "c_proj.weight": w_out,
        "c_proj.bias": b_out
    }

    true_attn = attn.MultiHeadAttention(config,weights)

    attn_in = torch.randn((config.seq_len,config.d_model))
    attn_in_packed = pack.pack_from_row(attn_in)

    mask = torch.tril(torch.ones((config.seq_len, config.seq_len)))
    mask2 = torch.triu(torch.ones((config.seq_len, config.seq_len)),diagonal=1)
    
    mask_packed = pack.pack_from_row(mask2)
    ex_mask_packed = pack.pack_from_row(mask)
    assert mask_packed.shape == (1,32768)

    qw_packed = pack.pack_from_row(qw.T)
    qb_packed = pack.expand_bias_head_row(qb,config.n_heads)

    kw_packed = pack.pack_from_row(kw.T)
    kb_packed = pack.expand_bias_head_row(kb,config.n_heads)

    vw_packed = pack.pack_from_row(vw.T)
    vb_packed = pack.expand_bias_head_col(vb,config.n_heads,128,64)

    w_out_packed = pack.pack_from_row(w_out.T)
    b_out_packed = pack.expand_bias(b_out)


    res = attn.attention_layer(attn_in_packed,
                               qw_packed,qb_packed,
                               kw_packed,kb_packed,
                               vw_packed,vb_packed,
                               w_out_packed,b_out_packed,
                               mask_packed,ex_mask_packed)

    exp = true_attn(attn_in,attn_in,attn_in,mask).detach().numpy()

    assert exp.shape == (128,768)

    print(f"res: {res[0][:10]}")
    print(f"exp: {exp[0][:10]}")

    res = pack.pack_tight(res)

    assert np.isclose(res.reshape(-1),exp.reshape(-1),atol=0.0009).all()

