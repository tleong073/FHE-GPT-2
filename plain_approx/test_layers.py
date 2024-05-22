import pytest
import numpy as np
import matrix_mul as matmul
import iterations as iter
import attn
import numpy.random as rand
import fold
import pack
import torch
import torch.nn as nn
import layers
import math
import attn


def fake_layernorm(A,W,B):
    mean = np.mean(A,axis=1)
    var = np.var(A,axis=1)
    div = np.sqrt(var)
    stnd = (A - mean.reshape(-1,1))/div.reshape(-1,1)
    return stnd*W +B

@pytest.mark.layernorm
def test_layernorm():
    A = rand.random((128,768))
    A_t = torch.tensor(A,device='cuda')

    W_data = np.ones((768,))
    W_tile = np.append(W_data,[np.zeros((2048-768,))])
    W = np.tile(W_tile,reps=16)
    assert W.shape[0] == 32768
    assert (W[:768] != 0).all()
    assert(W[768:2048] == 0).all()

    B_data = np.zeros((768,))
    B_tile = np.append(B_data,[np.zeros((2048-768,))])
    B = np.tile(B_tile,reps=16)

    # Pack into ciphertext format
    A_packed = pack.pack_from_row(A)

    assert (A_packed[0][:768] == A[0]).all()

    res = layers.layer_norm(A_packed,W,B,768,40298902)

    # Initialize torch primitives
    layer_norm = torch.nn.LayerNorm(768,bias=True)
    exp = None
    with torch.no_grad():
      layer_norm.weight = torch.nn.Parameter(torch.tensor(W_data,device='cuda'))
      layer_norm.bias = torch.nn.Parameter(torch.tensor(B_data,device='cuda'))
      exp = layer_norm(A_t).cpu().numpy()
    exp2 = fake_layernorm(A,W_data,B_data)
    print("RES: ",res[0][:10], math.sqrt(768))
    print("EXP: ",exp[0][:10])
    print(f"EXP2: {exp[0][:10]}")

    #assert np.isclose(exp.reshape(-1),res.reshape(-1)).all()


class Config:
    def __init__(self,d_model,d_hidden):
      self.d_model = d_model
      self.d_hidden = d_hidden
    

class FeedForward(torch.nn.Module):
    def __init__(self,config,weights):

        super(FeedForward,self).__init__()


        self.linear1 = nn.Linear(config.d_model,config.d_hidden)
        self.linear2 = nn.Linear(config.d_hidden,config.d_model)

        with torch.no_grad():
          self.linear1.weight = nn.Parameter(weights["w1"].T)
          self.linear1.bias = nn.Parameter(weights["b1"])

          self.linear2.weight = nn.Parameter(weights["w2"].T)
          self.linear2.bias = nn.Parameter(weights["b2"])


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

@pytest.mark.mlp
def test_mlp():
  A = rand.random((128,768))
  W1 = rand.random((768,3072))
  b1 = rand.random((3072,))

  W2 = rand.random((3072,768))
  b2 = rand.random((768,))

  g = np.ones((768,))
  b = np.zeros((768,))

  weights = {
     "w1": torch.tensor(W1,device='cuda'),
     "b1": torch.tensor(b1,device='cuda'),
     "w2": torch.tensor(W2,device='cuda'),
     "b2": torch.tensor(b2,device='cuda')
  }
  config = Config(768,3072)


  ff = FeedForward(config,weights)

  A_packed = pack.pack_from_row(A)
  
  W1_packed = pack.pack_from_row(W1.T)
  b1_packed = pack.expand_bias(b1)

  W2_packed = pack.pack_from_row(W2.T)
  b2_packed = pack.expand_bias(b2)

  gamma = pack.expand_bias(g)
  beta = pack.expand_bias(b)

  res = layers.mlp(A_packed,W1_packed,b1_packed,W2_packed,b2_packed,gamma,beta)
  #res = pack.pack_tight(res)
  #res = layers.layer_norm(res,gamma,beta,768,4.30307775e+15)
  #res = pack.pack_tight(res)

  exp = ff(torch.tensor(A,device='cuda'))
  layer_norm = torch.nn.LayerNorm(768,bias=True)
  with torch.no_grad():
    layer_norm.weight = torch.nn.Parameter(torch.tensor(g,device='cuda'))
    layer_norm.bias = torch.nn.Parameter(torch.tensor(b,device='cuda'))
  exp = layer_norm(exp).cpu().detach().numpy()
  #exp = exp.cpu().detach().numpy()

  print(f"RES: {res[0][:20]}")
  print(f"EXP: {exp[0][:20]}")
  

  assert np.isclose(res.reshape(-1),exp.reshape(-1),atol=0.001).all()










    
    

