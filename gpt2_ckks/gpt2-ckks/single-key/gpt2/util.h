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
#define LOGP 46
#define LOGQ 49
#define ENCODE_SCALE pow(2,LOGP)
#define BOOT_LEVEL 14 
#define TOTAL_LEVEL 35
#define THREAD_NUM 32

#define vec vector<double>
#define vvec vector<vector<double>>
#define vc vector<Ciphertext>

#define PRINT_CIPHER(X,str) printf(str,X.scale(),X.coeff_modulus_size())

// Macro used for test setup
#define INIT(X) EncryptionParameters params;\
PublicKey public_key;\
RelinKeys relin_keys;\
GaloisKeys gal_keys;\
SecretKey secret_key;\
vector<int> coeff_bit_vec;\
cout << "Setting Parameters" <<endl;\
params = EncryptionParameters(scheme_type::ckks);\
coeff_bit_vec.push_back(logq);\
for (int i = 0; i < remaining_level; i++) coeff_bit_vec.push_back(logp);\
for (int i = 0; i < boot_level; i++) coeff_bit_vec.push_back(logq);\
coeff_bit_vec.push_back(log_special_prime);\
size_t poly_modulus_degree = (size_t)(1 << logN);\
params.set_poly_modulus_degree(poly_modulus_degree);\
params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));\
size_t secret_key_hamming_weight = 192;\
params.set_secret_key_hamming_weight(secret_key_hamming_weight);\
double scale = pow(2.0, logp);\
SEALContext context(params);\
KeyGenerator keygen(context);\
keygen.create_public_key(public_key);\
cout << "Setting Parameters2" <<endl;\
secret_key = keygen.secret_key();\
keygen.create_relin_keys(relin_keys);\
vector<int> gal_steps_vector;\
for(int i=0; i<logN-1; i++) gal_steps_vector.push_back((1 << i));\
vector<int> rotation_kinds = {0,1,2,3,4,5,6,7,8,9,10,32640,31744\
,12288,16384,20480,24576,28672,32672,32704,32736,32,64,96,31872,32096,32320,32544,224\
,448,672,896,32765,32766,32767,32740,32747,32754,32761,14,21,28};\
for(int i=0; i<32768; i+=2048) rotation_kinds.push_back(i);\
for(auto rot: rotation_kinds)\
{\
	if(find(gal_steps_vector.begin(), gal_steps_vector.end(), rot) == gal_steps_vector.end()) gal_steps_vector.push_back(rot);\
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

void add_galois_keys(vector<double>&gal_steps_vector);

void init_bootstrap(Bootstrapper &bootstrapper,vector<int> &gal_steps_vector,int logn);

void bootstrap(Ciphertext &ctxt, Ciphertext &rtn, Bootstrapper &bootstrapper, Evaluator &evaluator);

void surefire_rotate(Ciphertext &cipher,int shift_amt, KeyGenerator &keygen,Evaluator &evaluator);
