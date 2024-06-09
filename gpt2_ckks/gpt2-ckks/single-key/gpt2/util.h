#pragma once
#include "seal/seal.h"
#include "MinicompFunc.h"
#include "func.h"
#include "PolyUpdate.h"
#include "Bootstrapper.h"
#include "tensor.h"
#include <omp.h>
#include <NTL/RR.h>

#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>


using namespace std;
using namespace seal;
using namespace minicomp;

// Other useful defines
#define LOGP 50
#define ENCODE_SCALE pow(2,LOGP)
#define vec vector<double>
#define vvec vector<vector<double>>
#define vc vector<Ciphertext>

// Macro used for test setup
#define INIT(X) EncryptionParameters params;\
PublicKey public_key;\
RelinKeys relin_keys;\
GaloisKeys gal_keys;\
SecretKey secret_key;\
vector<int> coeff_bit_vec;\
params = EncryptionParameters(scheme_type::ckks);\
coeff_bit_vec.push_back(logq);\
for (int i = 0; i < remaining_level; i++) coeff_bit_vec.push_back(logp);\
for (int i = 0; i < boot_level; i++) coeff_bit_vec.push_back(logq);\
coeff_bit_vec.push_back(log_special_prime);\
cout << "Setting Parameters" << endl;\
size_t poly_modulus_degree = (size_t)(1 << logN);\
params.set_poly_modulus_degree(poly_modulus_degree);\
params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));\
size_t secret_key_hamming_weight = 192;\
params.set_secret_key_hamming_weight(secret_key_hamming_weight);\
double scale = pow(2.0, logp);\
SEALContext context(params);\
KeyGenerator keygen(context);\
keygen.create_public_key(public_key);\
secret_key = keygen.secret_key();\
keygen.create_relin_keys(relin_keys);\
vector<int> gal_steps_vector;\
for(int i=0; i<logN-1; i++) gal_steps_vector.push_back((1 << i));\
vector<int> rotation_kinds = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,28672,\
6144,8192,10240,12288,14336,16384,18432,20480,22528,24576,26624,28672,30720,\
768,1536,2304,3072,3840,4608,5376,6144,6912,7680,8448,9216,9984,10752,11520,12288,13056,13824,14592,15360,16128,16896,17664,\
18432,19200,19968,20736,21504,22272,23040,23808,24576,25344,26112,26880,27648,28416,29184,29952,30720,31488,32256\
};\
gal_steps_vector.push_back(0);\
for(int i=0; i<32768; i+=128) gal_steps_vector.push_back(i);\
for(int i=0; i<logN-1; i++) gal_steps_vector.push_back(32768-(1 << i));\
for(auto rot: rotation_kinds)\
{\
	if(find(gal_steps_vector.begin(), gal_steps_vector.end(), rot) == gal_steps_vector.end()) gal_steps_vector.push_back(rot);\
	if(rot!=0 && find(gal_steps_vector.begin(), gal_steps_vector.end(), 32768-rot) == gal_steps_vector.end()) gal_steps_vector.push_back(32768-rot);\
}\
keygen.create_galois_keys(gal_steps_vector,gal_keys);\
CKKSEncoder encoder(context);\
Encryptor encryptor(context, public_key);\
Evaluator evaluator(context, encoder);\
Decryptor decryptor(context, secret_key)


int round_to_2(double x);

void rotate_inplace(Ciphertext &cipher_in, int steps, Evaluator &evaluator, GaloisKeys &gal_keys);

void rotate_vec(const Ciphertext &cipher_in, Ciphertext & cipher_out, int steps, Evaluator &evaluator, GaloisKeys &gal_keys);

void fakeBootstrap(Ciphertext &input,Ciphertext &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void init_output(int num_ciphers,vector<Ciphertext> &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void mask_out(Ciphertext &cipher,Ciphertext &out, int start,int length,CKKSEncoder &encoder,Evaluator &evaluator,RelinKeys &relin_keys);

void pack_plain_row(vector<vector<double>> &v,int rows,int row_size,vector<vector<double>> &out);