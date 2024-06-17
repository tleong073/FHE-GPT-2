#include "approx.h"

// Implements kv caching augmentation in-place
void augment_value_row(vc& A, vc& cached_val,int padded_row_size,int idx,
	CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys)
{
    vec mask(32768,1.0);
    fill(mask.begin()+(idx*padded_row_size),mask.begin()+((idx+1)*padded_row_size),0.0);

    Ciphertext masked_out;
    Plaintext plain;

    encoder.encode(mask,ENCODE_SCALE,plain);

    int i;
    for(i=0;i<A.size();i++){
        evaluator.multiply_plain_inplace(A[i],plain);
        evaluator.rescale_to_next_inplace(A[i]);
        evaluator.add_inplace_reduced_error(A[i],cached_val[i]);
    }

}

void augment_value_col(vc& A,vc& cached_val,int padded_row_size,int idx,
	CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys)
{
    vec mask(32768,1.0);
    int i;
    for(i=0;i<padded_row_size/2;i++)
        mask[i*padded_row_size+idx]=0.0;

    Plaintext plain;
    encoder.encode(mask,ENCODE_SCALE,plain);
    
    for(i=0;i<A.size();i++){
        evaluator.multiply_plain_inplace(cached_val[i],plain);
        evaluator.rotate_vector_inplace(A[i],idx,gal_keys);
        evaluator.add_inplace_reduced_error(A[i],cached_val[i]);
    }
}