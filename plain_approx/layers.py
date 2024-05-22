import fold
import iterations as iter
import matrix_mul as matmul
import pack
import math
import numpy as np
import numpy.random as rand
import poly
import torch


def round_to_2(x):
    return math.pow(2,math.ceil(math.log(x,2)))

def mask_out(arr,bit_range):
    #bit_range is specified as (start,len)
    mask = np.zeros_like(arr,dtype=float)
    for i in range(bit_range[0],bit_range[0]+bit_range[1]):
        mask[i] = 1.0
    out = arr * mask
    return out

# Compute layer norm. Assumes row-packed with 128 rows of size 768
def layer_norm(A,gamma,beta,row_size,newton_init_val):

    rounded_row_size = int(round_to_2(row_size))

    # Assume output is packed for bootstrap
    out = np.zeros((3,32768))

    sums = np.zeros_like(A)
    z = np.zeros_like(A)
    y = np.zeros_like(A)
    print(A.shape)

    mask = np.zeros((32768))
    ones = np.pad(np.ones(row_size),(0,32768-row_size),'constant')


    for i in range(16):
        mask += ones
        ones = np.roll(ones,rounded_row_size*2)

    # Assume packed as (128,768) in pre-fold format
    for i in range(A.shape[0]):
        combined = A[i] + np.roll(A[i], rounded_row_size)
        sums[i] = fold.quickSum(combined,rounded_row_size*2)

        if i == 0:
            print(f"folded: {sums[i][:10]} {A[i][:768].sum()}")
        # Compute z = n x 768  - sum
        z[i] = row_size * A[i]
        z[i] = z[i] - sums[i]
        if i == 0:
            print(f"postsqrt EXP: {1/np.sqrt(np.square(z[i][:768]).sum())}")
            print(f"EXP: {math.sqrt(row_size)*z[i][:10]/np.sqrt(np.square(z[i][:768]).sum())}")

        # Compute y = square(z)
        y[i] = z[i]*z[i]

        # Mask out to convert to prefold format
        y[i] = y[i] * mask
        assert (y[i][768:2048] == 0.0).all()

        pre_sum = y[i]
        # Fold again
        y[i] = y[i] + np.roll(y[i],rounded_row_size)
        y[i] = fold.quickSum(y[i],rounded_row_size*2)

        if i == 0:
            print(f"folded2: {y[i][:10]} {pre_sum[:768].sum()}")

        if i == 0:
            print(f"Pre Newton res: {rounded_row_size} {y[i][:10]} {1/np.sqrt(y[i][:10])}")
        
        # Compute inv sqrt w/ Newton method
        y[i] = iter.newton_iteration(y[i]+1,newton_init_val,13)
        #y[i] = np.nan_to_num(y[i],nan=0.0,posinf=0.0,neginf=0.0)
        #y[i] *= mask

        if i == 0:
            print(f"Post Newton res: {y[i][:10]} ")
        # Compute y = z*y
        y[i] = z[i] * y[i]

        y[i] = ((y[i]*gamma) * math.sqrt(row_size)) + beta
        if i == 0:
            print(f"Post norm res: {y[i][:10]} ")
    
    return y#pack.pack_tight(y)

# Perform Feed Forward Steps
def mlp(A,W1,b1,W2,b2):
    
    assert A.shape == (8,32768)
    assert W1.shape == (192,32768) and b1.shape[0] == 32768
    assert W2.shape == (192,32768) and b2.shape[0] == 32768

    # Initial hidden layer for MLP () x ()
    hidden_in = matmul.generic_matrix_mul(A,W1,np.zeros((32768,)),128,768,768,3072)

    gelu = torch.nn.GELU(approximate='tanh')
    # Add Bias and run GELU on matrices
    for i in range(hidden_in.shape[0]):
        hidden_in[i] += b1
        hidden_in[i] = gelu(torch.tensor(hidden_in[i])).numpy()

    # One more dense layer (128,3072) x (3072,768)
    hidden_out = matmul.generic_matrix_mul(hidden_in,W2,np.zeros((32768,)),128,3072,3072,768)

    # Bias
    for i in range(hidden_out.shape[0]):
        hidden_out[i] += b2
    assert hidden_out.shape == (8,32768)

    return hidden_out

