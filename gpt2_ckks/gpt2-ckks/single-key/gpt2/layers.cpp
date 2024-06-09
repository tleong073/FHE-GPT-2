#include "approx.h"

void FeedForwardLayer( vector<Ciphertext>&A,vector<Ciphertext> W1,Ciphertext b1,vector<Ciphertext> W2,Ciphertext b2,vector<Ciphertext> &outputs,
	 CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys)

{
    Ciphertext cipher;
    vector<Ciphertext> hidden_input;

    init_output(192,hidden_input,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    // Initial dense layer to transform into ff_hidden size
    row_matrix_multiplication_seal(A,W1,b1,hidden_input,128,768,768,3072,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    int i;
    for(i=0;i<hidden_input.size();i++){
        gelu(hidden_input[i],hidden_input[i],encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    }

    // Output dense layer
    row_matrix_multiplication_seal(hidden_input,W2,b2,outputs,128,3072,3072,768,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);


}

void attentionLayer( vc&A,vc&qw, Ciphertext qb,vc&kw,Ciphertext kb,vc&vw,Ciphertext vb,vc&w_out,Ciphertext &b_out,Ciphertext mask,vector<vc>& kv_cache,vc &outputs, int rows, int cols, int idx,
						CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys)
{
    
    int i;

    vc Q,K,V;

    init_output(8,Q,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    init_output(8,K,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    init_output(8,V,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    // Q,K,V projection
    attn_proj_row_seal(A,qw,qb,Q,128,768,768,768,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    attn_proj_row_seal(A,kw,kb,K,128,768,768,768,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    attn_proj_col_seal(A,vw,vb,V,128,768,768,768,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    // Instead of NULL, use cache size to case on existence of cache
    if(kv_cache.size() > 0){
        augment_value_row(K,kv_cache[0],128,idx,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        augment_value_col(V,kv_cache[1],256,idx,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        // Update current cache
        kv_cache[0] = K;
        kv_cache[1] = V;
    }

    vc QKt;
    init_output(12,QKt,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    qk_matmul(Q,K,QKt,128,768,768,128,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    // Apply mask in batched case
    if(kv_cache.size() == 0){
        for(i=0;i<QKt.size();i++){
            evaluator.add_inplace_reduced_error(QKt[i],mask);
        }
    }

    // Apply softmax
    for(i=0;i<QKt.size();i++)
        compute_softmax(QKt[i],6,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    
    vc pre_out;
    init_output(8,QKt,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    sv_matmul(QKt,V,pre_out,128,128,128,768,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    row_matrix_multiplication_seal(pre_out,w_out,b_out,outputs,128,768,768,768,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
}
