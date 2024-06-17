#include "pack.h"
#include "util.h"

#include <algorithm>
#include <bits/stdc++.h>



// Assume pre initialized output
void pack_tight(vector<Ciphertext> &input,vector<Ciphertext> &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys) {
                            // TODO: Implement this with precomputed masks
                            Plaintext plain,tmp;
                            Ciphertext cipher,tmp_cipher,masked_out;

                            bool flag = false;
                            int global_idx = 0,leftover=0;

                            for(int i = 0; i< 8;i++) {
                                printf("Pack tight: %d\n",i);
                                flag=false;
                                for(int j = 0; j< 16;j++) {
                                    if( (j == 15 && flag) || (global_idx == 98304))
                                        break;
                                    
                                    mask_out(input[i],masked_out,0,768,encoder,evaluator,relin_keys);
                                    printf("Pack tight after mask: %d %d %d %zu\n",i,j,global_idx%32768,masked_out.coeff_modulus_size());
                                    //evaluator.rotate_vector_inplace(masked_out, 32768-(global_idx % 32768),gal_keys);
                                    rotate_inplace(masked_out,-(global_idx % 32768),evaluator,gal_keys);
                                    decrypt_and_print_and_max_round(masked_out,decryptor,encoder,1.0,0);
                                    evaluator.add_inplace_reduced_error(output[global_idx / 32768],masked_out);

                                    global_idx += 768;
                                    evaluator.rotate_vector_inplace(input[i],2048,gal_keys);
                                    printf("Pack tight after rotate: %d %d\n",i,j);
                                    
                                    leftover = (32768 - (global_idx % 32768));
                                    if(leftover < 768) {

                                        mask_out(input[i],masked_out,0,leftover,encoder,evaluator,relin_keys);
                                        //evaluator.rotate_vector_inplace(masked_out,32768-(global_idx % 32768),gal_keys);
                                        rotate_inplace(masked_out,-(global_idx % 32768),evaluator,gal_keys);
                                        evaluator.add_inplace_reduced_error(output[global_idx / 32768],masked_out);
                                        global_idx += leftover;

                                        mask_out(input[i],masked_out,leftover,768-leftover,encoder,evaluator,relin_keys);
                                        //evaluator.rotate_vector_inplace(masked_out,leftover,gal_keys);
                                        rotate_inplace(masked_out,leftover,evaluator,gal_keys);
                                        evaluator.add_inplace_reduced_error(output[global_idx / 32768],masked_out);
                                        global_idx += 768 - leftover;
                                        
                                        //evaluator.rotate_vector_inplace(input[i],2048,gal_keys);
                                        rotate_inplace(input[i],2048,evaluator,gal_keys);
                                        flag=true;
                                    }

                                }
                            }

                            return;
                        }

/*
def unpack_tight(arr):
    assert len(arr) == 3
    x = np.zeros((8,32768))
    src_idx = 0
    dst_idx = 0
    while src_idx < 32768*3:
        print(f"dst_idx: {dst_idx} src_idx: {src_idx}")
        masked_out = mask_out(arr[src_idx // 32768],(src_idx % 32768,768))

        shift_amt = (dst_idx % 32768) - (src_idx % 32768)
        x[dst_idx // 32768] += np.roll(masked_out,shift_amt)
        
        src_idx += 768
        dst_idx += 2048

        leftover = 32768 - (src_idx % 32768)
        if leftover < 768: 
            # Put remaining chunks into array
            print(dst_idx // 32768,src_idx // 32768,dst_idx % 32768)
            masked_out = mask_out(arr[src_idx // 32768],(src_idx % 32768,leftover))
            shift_amt = (dst_idx % 32768) - (src_idx % 32768)
            x[dst_idx // 32768] += np.roll(masked_out,shift_amt)

            src_idx += leftover
            dst_idx += leftover

            # Fill in remaining row
            print(dst_idx // 32768,src_idx // 32768,dst_idx % 32768)
            masked_out = mask_out(arr[src_idx // 32768],(src_idx % 32768,768-leftover))
            shift_amt = (dst_idx % 32768) - (src_idx % 32768)
            x[dst_idx // 32768] += np.roll(masked_out,shift_amt)

            src_idx += 768-leftover
            dst_idx += 2048-leftover
    return x
*/
// Assume outputs are preformatted
void unpack_tight(vector<Ciphertext> &input,vector<Ciphertext> &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys)
{
    int src_idx = 0;
    int dst_idx = 0;
    int shift_amt;

    Ciphertext cipher,rolled;

    while (src_idx < 32768 * 3) {
        std::cout << "dst_idx: " << dst_idx << " src_idx: " << src_idx << std::endl;
        //std::vector<double> masked_out = mask_out(arr[src_idx / 32768], {src_idx % 32768, 768});
        mask_out(input[src_idx/32768],cipher,src_idx % 32768,768,encoder,evaluator,relin_keys);
        shift_amt =  (src_idx % 32768) - (dst_idx % 32768);
        evaluator.rotate_vector(cipher,shift_amt,gal_keys,rolled);

        evaluator.add_inplace_reduced_error(output[dst_idx /32768],rolled);

        src_idx += 768;
        dst_idx += 2048;

        int leftover = 32768 - (src_idx % 32768);
        if (leftover < 768) {
            std::cout << dst_idx / 32768 << " " << src_idx / 32768 << " " << dst_idx % 32768 << std::endl;

            mask_out(input[src_idx/32768],cipher,src_idx % 32768,leftover,encoder,evaluator,relin_keys);
            
            shift_amt =  (src_idx % 32768) - (dst_idx % 32768);
            
            evaluator.rotate_vector(cipher,shift_amt,gal_keys,rolled);
            evaluator.add_inplace_reduced_error(output[dst_idx / 32768],rolled);

            src_idx += leftover;
            dst_idx += leftover;

            std::cout << dst_idx / 32768 << " " << src_idx / 32768 << " " << dst_idx % 32768 << std::endl;
            mask_out(input[src_idx/32768],cipher,src_idx % 32768,2048-leftover,encoder,evaluator,relin_keys);
            
            shift_amt =  (src_idx % 32768) - (dst_idx % 32768);
            
            evaluator.rotate_vector(cipher,shift_amt,gal_keys,rolled);
            evaluator.add_inplace_reduced_error(output[dst_idx / 32768],rolled);

            src_idx += 768 - leftover;
            dst_idx += 2048 - leftover;
        }
    }

    return;
}

// Only works in 2-D matrices. Assume out is prepared as packed 0 ciphertexts
void pack_from_row(vector<vector<double>> &input, vector<Ciphertext> &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys) {
                            
        // Dimensions required for calculation
        int rows = input.size();
        int cols = input[0].size();
        int chunk_size = round_to_2(cols)*2;
        int chunks_per_cipher = 32768 / chunk_size;
        int num_ciphers = max(1,(rows*chunk_size)/32768);
        int chunk_offset,out_idx,rot_amt;
        printf("rows: %d cols: %d chunk_size: %d per_cipher: %d\n",rows,cols,chunk_size,chunks_per_cipher);
        Plaintext plain,tmp_padded;
        Ciphertext cipher,padded,rolled;
        
        vector<vector<double>> plain_out(num_ciphers,vector<double>(32768,0));

        pack_plain_row(input,rows,cols,plain_out);
        int i;
        for(i=0;i<num_ciphers;i++){
            encoder.encode(plain_out[i],ENCODE_SCALE,plain);
            encryptor.encrypt(plain,cipher);
            output.push_back(cipher);
        }
        
        return;
}

vector<double> repeat(vector<double> &input,int times) {
    vector<double> result(input.size() * times);

    for(size_t rep = 0;rep < times;++rep){
        copy(input.begin(),input.end(),next(result.begin(),rep*input.size()));
    }
    return result;
}

void expand_bias(vector<double> &input,Ciphertext &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys) {
    // TODO: Implement this
    // Dimensions
    int sz = input.size();
    int sz_rounded = round_to_2(sz) * 2;
    
    vector<double> tile(input.begin(),input.end());
    tile.resize(sz_rounded-sz);

    vector<double>res = repeat(tile,32768/sz_rounded);


    Plaintext plain;
    encoder.encode(res,ENCODE_SCALE,plain);
    encryptor.encrypt(plain,output);

    return;
}

// Assume preformated output
void expand_bias_head_row(vector<double> &input,vector<Ciphertext> &output,int heads, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys) {

    // Dimensions
    int sz = input.size() / heads;

    vector<double> res;
    vector<double> tmp(2*sz,0.0);
    for(int i = 0; i<heads;i++) {
        copy(input.begin()+i*sz,input.begin()+(i+1)*sz,tmp.begin());
        res = repeat(tmp,(16384/(sz*2)));
        // Zero Pad
        res.resize(32768);
        tmp.clear();
    }

    return;
}

// Assume preformatted output
void expand_bias_head_col(vector<double> &input,vector<Ciphertext> &output, int heads, int rows, int cols, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys) {
    
    // TODO: Implement this
    vector<double>res(32768,0);
    vector<double>tmp;
    Plaintext plain;
    for(int i = 0; i<heads;i++){
        for(int j = 0; j<cols;j++){
            
            tmp = vector<double>(rows,input[i*cols+j]);
            copy(tmp.begin(),tmp.end(),next(res.begin(),j*rows*2));

            encoder.encode(res,ENCODE_SCALE,plain);
            evaluator.add_plain_inplace(output[i],plain);

            tmp.clear();
        }
    }
    return;
}

// Restore head packing into row format
void pack_heads(vector<Ciphertext> &input, vector<vector<double>> &output,int heads, int num_ciphers,int num_rows,int row_size, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys) {
                            //TODO: implement
    
}