import numpy as np
import math

def mask_out(arr,bit_range):
    #bit_range is specified as (start,len)
    mask = np.zeros_like(arr,dtype=float)
    for i in range(bit_range[0],bit_range[0]+bit_range[1]):
        mask[i] = 1.0
    out = arr * mask
    return out

def pack_tight(arr):
    x = np.zeros((3,32768))
    assert arr.shape[0] == 8
    global_idx = 0
    for i in range(8):
        flag = False
        for j in range(16):

            if j == 15 and flag:
                break
            if global_idx == 98304:
                break

            masked_out = mask_out(arr[i],(0,768))
            x[global_idx // 32768] += np.roll(masked_out,global_idx % 32768)
            
            global_idx += 768
            arr[i] = np.roll(arr[i],-2048)

            leftover = (32768 - (global_idx % 32768))
            if leftover < 768:
                
                masked_out = mask_out(arr[i],(0,leftover))
                
                x[global_idx // 32768] += np.roll(masked_out, global_idx % 32768)
                global_idx += leftover

                masked_out = mask_out(arr[i],(leftover,768-leftover))

                #print(np.roll(masked_out,-leftover),leftover,global_idx % 32768,global_idx // 32768,i,j)
                rot = np.roll(masked_out,-leftover)
                x[global_idx // 32768] += rot
                global_idx += 768 - leftover

                arr[i] = np.roll(arr[i], -2048)
                flag=True
            #print(arr[i+1][:10])
    return x

def round_to_2(x):
    return math.pow(2,math.ceil(math.log(x,2)))

# Only works on 2-D matrices
def pack_from_row(A):
    rows,cols = A.shape
    chunk_size = int(round_to_2(cols)*2)
    chunks_per_cipher = 32768 // chunk_size
    num_ciphers = math.ceil((rows*chunk_size)/32768)

    out = np.zeros((num_ciphers,32768))
    #print(chunk_size,chunks_per_cipher,num_ciphers)
    global_idx = 0
    for i in range(rows):
        padded = np.pad(A[i], (0,32768-cols),'constant')
        #print(padded.shape,out.shape,type(global_idx // 32768))
        chunk_offset = int((global_idx // chunk_size) % chunks_per_cipher)
        #print(chunk_offset)
        rolled = np.roll(padded,chunk_size*chunk_offset)
        #print(rolled.shape)
        out_idx = int(global_idx // 32768)
        out[out_idx] += rolled
        global_idx += chunk_size 
    
    return out

# Expands from |bias| to |bias|0000|bias|0000|.... 
def expand_bias(B):
    size = B.shape[0]
    size_rounded = int(round_to_2(size)) *2
    W_tile = np.append(B,[np.zeros((size_rounded-size,))]) 
    
    return np.tile(W_tile,reps = (32768 // size_rounded))

#Special packing for Q,K biases which are split into heads
def expand_bias_head_row(B,heads):
    assert B.shape[0] % heads == 0 # Hidden dim should be divisble by heads
    size = B.shape[0] // heads 

    out = np.zeros((heads,32768))
    
    # Bias should be split like:  Head1Bias | Head2Bias | Head3Bias |.....
    for i in range(heads):
        W_tile = np.append(B[i*size:(i+1)*size],[np.zeros((size,))])
        out[i] = np.append(np.tile(W_tile,reps = (16384 // (size*2))),[np.zeros((16384,))])
    
    return out

#Special packing for V bias which is split into heads and col packed
# Assumes V is rowpacked with dims (rows,cols)
def expand_bias_head_col(B,heads,rows,cols):
    assert B.shape[0] % heads == 0 # Hidden dim should be divisble by heads

    out = np.zeros((heads,32768))
    
    # Bias should be split like:  Head1Bias | Head2Bias | Head3Bias |.....
    # We require it to be like: h1b1|h2b1|h3b1... || h1b2|h2b2|h3b2...|| ...
    for i in range(heads):
        for j in range(cols):
            tmp = np.append(np.repeat(B[i*cols + j],rows),[np.zeros((32768-rows,))])
            out[i] += np.roll(tmp,j*rows*2)
    
    return out

# Assume heads are packed into 12 ciphers
def unpack_heads(heads,num_ciphers,num_rows,row_size):
    #assert heads.shape == (12,32768)

    out = np.zeros((num_ciphers,num_rows,row_size))

    for i in range(num_ciphers):
        for j in range(num_rows):
            out[i][j] = heads[i][row_size*2*j:row_size*2*j+row_size]
    
    return out

def pack_heads(A,num_ciphers,num_rows,row_size):
    out = np.zeros((num_ciphers,32768))
    for i in range(num_ciphers):
        for j in range(num_rows):
            padded = np.pad(A[i][j], (0,32768-row_size),'constant')
            out[i] += np.roll(padded, (row_size*2)*j)
    return out