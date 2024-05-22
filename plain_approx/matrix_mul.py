import numpy as np
import jax.numpy as jnp
import fold
import math


def round_to_2(x):
    return math.pow(2,math.ceil(math.log(x,2)))

def mask_out(arr,bit_range):
    #bit_range is specified as (start,len)
    mask = np.zeros_like(arr,dtype=float)
    for i in range(bit_range[0],bit_range[0]+bit_range[1]):
        mask[i] = 1.0
    out = arr * mask
    return out


def row_matrix_mul(input_matrix,weight_rows):
    output_rows = []
    for i in range(len(input_matrix)):
        acc = np.zeros_like(weight_rows[0])
        row = input_matrix[i]
        for j in range(len(row)):
            acc += row[j] * weight_rows[j]
        output_rows.append(acc)
    return np.array(output_rows)


def col_matrix_mul(left_input,right_input):
    output_rows = []
    for left in left_input:
        acc = []
        for right in right_input:
            prod = left * right
            prod = fold.quickSum(prod,len(prod))
            acc.append(prod) 
        output_rows.append(acc)
    return np.array(output_rows)

def diagonal_to_row(matrix):
    output_rows = []
    eyes = np.identity(len(matrix))
    for i in range(len(matrix)):
        acc = np.zeros_like(matrix[i][0],dtype=float)
        for j in range(len(matrix[i])):
            acc += matrix[i][j] * eyes[j]
        output_rows.append(acc)
    return np.array(output_rows)

# A*W, which is really row-wise AW^T
def generic_matrix_mul(A,W,B,A_rows,A_cols,W_rows,W_cols):


    
    W_rows_rounded = int(round_to_2(W_rows))
    W_cols_rounded = int(round_to_2(W_cols))

    # Dims
    chunk_size = W_rows_rounded*2
    chunks = int(32768 // chunk_size)

    out_chunk_size = W_cols_rounded*2
    #print(f"Generic matmul: {chunk_size} {chunks} {A_rows} {W_cols}")


    out = np.zeros(((A_rows * out_chunk_size)//32768,32768))
    
    for i in range(A.shape[0]):
        for j in range(W.shape[0]):
            for rots in range(chunks):
                rolled = np.roll(W[j],-(rots*chunk_size))
                res1 = A[i] * rolled

                # Format for fold
                #print("res1", (res1[1024:2048]==0).all())
                res = res1 + np.roll(res1,W_rows_rounded)
                folded = fold.quickSum(res,W_rows_rounded*2)
                
                for pos in range(chunks):
                    row = i * chunks + pos
                    col = j * chunks + ((rots + pos) % chunks)

                    # Mask out real chunk
                    masked_out = mask_out(folded,(pos*chunk_size,1))
                    
                    # Determine which cipher,chunk and offset to place it in
                    abs_pos = row * W_cols + col

                    cipher = (row*out_chunk_size) // 32768
                    chunk = ((row*out_chunk_size) % 32768) // out_chunk_size
                    chunk_offset = col 
                    
                    desired_location = chunk * out_chunk_size + chunk_offset                    
                    shift_amt = desired_location - pos*chunk_size
                    rolled = np.roll(masked_out, shift_amt)

                    if cipher == 0 and chunk == 1:
                        pre_roll = np.roll(masked_out,-pos*chunk_size)
                        #print("HERE ",i,j,rots,pos,chunk,chunk_offset)

                    # Rotate and Add
                   
                    #print(f"Rolled:  {cipher} {rolled.shape}")
                    #print(i,j,rots,pos,head_row,head_col, abs_pos,desired_location)

                    out[cipher] += rolled
    for i in range(len(out)):
        out[i] += B
    return out

