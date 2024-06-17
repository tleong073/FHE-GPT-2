#include "util.h"
#include "pack.h"
#include "Bootstrapper.h"

// Matrix related functions

void col_matrix_multiplication_seal( vector<TensorCipher> &left_inputs, vector<TensorCipher> &right_inputs, vector<TensorCipher> &outputs, vector<double> bias,
	int rows, int cols, Config &config, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);


void decrypt_and_print(const Ciphertext &cipher, Decryptor &decryptor, CKKSEncoder &encoder,int n);

void row_matrix_multiplication_seal( vector<Ciphertext> &left_inputs, vector<Ciphertext> &weights,Ciphertext bias, vector<Ciphertext> &outputs,
	int A_rows, int A_cols,int W_rows,int W_cols, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);

void attn_proj_row_seal( vector<Ciphertext> &left_inputs, vector<Ciphertext> &weights,Ciphertext bias, vector<Ciphertext> &outputs,
	int A_rows, int A_cols,int W_rows,int W_cols,KeyGenerator &keygen, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);

void attn_proj_col_seal( vector<Ciphertext> &left_inputs, vector<Ciphertext> &weights,Ciphertext bias, vector<Ciphertext> &outputs,
	int A_rows, int A_cols,int W_rows,int W_cols, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);

void qk_matmul( vector<Ciphertext> &Q, vector<Ciphertext> &K, vector<Ciphertext> &outputs,
	int A_rows, int A_cols,int W_rows,int W_cols, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);

void sv_matmul( vector<Ciphertext> &S, vector<Ciphertext> &V, vector<Ciphertext> &outputs,
	int A_rows, int A_cols,int W_rows,int W_cols, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);

// Non-linear poly function evaluation

void compute_sign_f(Ciphertext &input,Ciphertext &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void compute_sign_g(Ciphertext &input,Ciphertext &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void sign_function(TensorCipher &inputs,TensorCipher &outputs, int df,int dg, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void compute_gelu_p(Ciphertext &input,Ciphertext &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void compute_gelu_q(Ciphertext &input,Ciphertext &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void gelu(Ciphertext &inputs,Ciphertext &outputs, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void compute_exp(Ciphertext &input,Ciphertext &output,int r, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

// Non-linear iteration function evaluation
void compute_inverse(Ciphertext &input,Ciphertext &output,int iters, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void taylor_expand(Ciphertext &input,Ciphertext &output,int iters,double guess, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void compute_inv_sqrt(Ciphertext &input,Ciphertext &output,int iters,double guess, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

// Folding functions

void quickSum(Ciphertext &input,Ciphertext &output,int n, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void computeMax(Ciphertext &input1,Ciphertext &input2,Ciphertext &output, Bootstrapper &bootstraper, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void quickMax(Ciphertext &input,Ciphertext &output,int n, Bootstrapper &bootstrapper,CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

// Compound Non-linear functions
void compute_softmax(Ciphertext &input,int r,Bootstrapper &bootstrapper, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void compute_smax(Ciphertext &input,int r,int gamma, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void compute_layernorm(Ciphertext &input,Ciphertext &output,vector<double> gamma,vector<double> beta,int row_size, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);


// Full Layers
void FeedForwardLayer( vector<Ciphertext>&A,vector<Ciphertext> W1,vector<Ciphertext> b1,vector<Ciphertext> W2,vector<Ciphertext> b2,vector<Ciphertext> &outputs,
	 CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);

void attentionLayer( vc&A,vc&qw, Ciphertext qb,vc&kw,Ciphertext kb,vc&vw,Ciphertext vb,vc&w_out,Ciphertext &b_out,Ciphertext mask,vector<vc>& kv_cache,vc &outputs, int rows, int cols, int idx,
						CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);

// KV cache optimizations
void augment_value_row(vc& A, vc& cached_val,int padded_row_size,int idx,
	CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);

void augment_value_col(vc& A, vc& cached_val,int padded_row_size,int idx,
	CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);