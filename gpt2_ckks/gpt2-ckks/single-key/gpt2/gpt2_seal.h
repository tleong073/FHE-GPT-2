#include "approx.h"



void FeedForwardLayer( vector<TensorCipher>&inputs,vector<TensorCipher> &outputs, int rows, int cols, Config &config,
	 CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);

void attentionLayer( vector<TensorCipher>&inputs,vector<TensorCipher> &outputs, int rows, int cols, Config &config,
	 CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);