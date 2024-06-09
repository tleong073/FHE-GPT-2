#pragma once
#include "seal/seal.h"
#include "tensor.h"

using namespace std;
using namespace seal;

void pack_tight(vector<Ciphertext> &input,vector<Ciphertext> &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void pack_from_row(vector<vector<double>> &input, vector<Ciphertext> &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void expand_bias(vector<double> &input,Ciphertext &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void expand_bias_head_row(vector<double> &input,vector<Ciphertext> &output,int heads, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void expand_bias_head_col(vector<double> &input,vector<Ciphertext> &output, int heads, int rows, int cols, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void unpack_heads(vector<vector<double>> &input, vector<Ciphertext> &output,int heads, int num_ciphers,int num_rows,int row_size, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);