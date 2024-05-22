#pragma	once
#include "seal/seal.h"
#include "MinicompFunc.h"
#include "func.h"
#include "PolyUpdate.h"
#include "Bootstrapper.h"
#include <omp.h>
#include <NTL/RR.h>

#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>


using namespace std;
using namespace seal;
using namespace minicomp;


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
keygen.create_galois_keys(gal_keys);\
CKKSEncoder encoder(context);\
Encryptor encryptor(context, public_key);\
Evaluator evaluator(context, encoder);\
Decryptor decryptor(context, secret_key)


#define GET_LEVEL(x) context.get_context_data(x.parms_id())->chain_index()

class TensorCipher
{
private:
	int k_;		// k: gap
	int h_;		// w: height
	int w_;		// w: width
	int c_;		// c: number of channels
	int t_;		// t: \lfloor c/k^2 \rfloor
	int p_;		// p: 2^log2(nt/k^2hwt)
	int logn_;
	Ciphertext cipher_;

public:
	TensorCipher();
	TensorCipher(Ciphertext cipher);
	TensorCipher(int logn, int k, int h, int w, int c, int t, int p, vector<double> data, Encryptor &encryptor, CKKSEncoder &encoder, int logp); 	// data vector contains hxwxc real numbers. 
	TensorCipher(int logn, int k, int h, int w, int c, int t, int p, Ciphertext cipher);
	int k() const;
    int h() const;
    int w() const;
	int c() const;
	int t() const;
	int p() const;
    int logn() const;
	Ciphertext cipher() const;
	void set_ciphertext(Ciphertext cipher);
	void print_parms();
};

class Config
{
public:
	// SEAL and bootstrapping setting
	long boundary_K;
	long boot_deg;
    long scale_factor;
    long inverse_deg; 
	long logN;
	long loge; 
	long logn;		// full slots
	long logn_1;	// sparse slots
	long logn_2;
	long logn_3;
	int logp;
	int logq;
	int log_special_prime;
    int log_integer_part;
	int remaining_level;
	int boot_level; 
	int total_level;
	Config();
	Config(long boundary_K,
	long boot_deg,
    long scale_factor,
    long inverse_deg,
	long logN,
	long loge,
	long logn,
	long logn_1,
	long logn_2,
	long logn_3,
	int logp,
	int logq,
	int log_special_prime,
    int log_integer_part,
	int remaining_level,
	int boot_level,
	int total_level);
};

class CKKSObjects
{
public:
	CKKSEncoder *encoder;
	Encryptor *encryptor;
	Evaluator *evaluator;
	Decryptor *decryptor;
	GaloisKeys * gal_keys;
	RelinKeys *relin_keys;
	CKKSObjects();
	CKKSObjects(
	 CKKSEncoder &encoder,
	 Encryptor &encryptor,
	 Decryptor &decryptor,
	 Evaluator &evaluator,
	 GaloisKeys &gal_keys,
	 RelinKeys &relin_keys);
};

class ChebyPoly
{
public:
	// Assume basis consists of even-numbered chebyshev polynomials + 1
	// [T0,T1,T2,T4,T8]
	vector<double> _coeffs;
	ChebyPoly *_q;
	ChebyPoly *_r;
	bool _isLeaf;
	bool _isQ;
	ChebyPoly();
	ChebyPoly(vector<vector<double>> coeffs, ChebyPoly *q, ChebyPoly *r, bool isLeaf, bool isQ);
	void BuildChebyBasis(CKKSObjects ckks_obs,TensorCipher &input, vector<TensorCipher> &chebyBasis, int n);
	void Eval(CKKSObjects ckks_obs,vector<TensorCipher> chebyBasis, TensorCipher &output);
};


// Matrix related functions

void col_matrix_multiplication_seal( vector<TensorCipher> &left_inputs, vector<TensorCipher> &right_inputs, vector<TensorCipher> &outputs, vector<double> bias,
	int rows, int cols, Config &config, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);

void memory_save_rotate(const Ciphertext &cipher_in, Ciphertext &cipher_out, int steps, Evaluator &evaluator, GaloisKeys &gal_keys);

void decrypt_and_print(const Ciphertext &cipher, Decryptor &decryptor, CKKSEncoder &encoder,int n);

void row_matrix_multiplication_seal( vector<vector<TensorCipher>> &left_inputs, vector<vector<double>> &weights, vector<TensorCipher> &outputs, vector<double> bias,
	int rows, int cols, Config &config, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);

void diagonal_to_row_matrix_seal( vector<TensorCipher>&inputs,vector<TensorCipher> &outputs, int rows, int cols, Config &config,
	 CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);

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

void gelu(TensorCipher &inputs,TensorCipher &outputs, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

// Non-linear iteration function evaluation
void compute_inverse(Ciphertext &input,Ciphertext &output,int iters, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void compute_inv_sqrt(Ciphertext &input,Ciphertext &output,int iters,double guess, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void compute_exp(Ciphertext &input,Ciphertext &output,int r, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

// Folding functions

void quickSum(TensorCipher &input,TensorCipher &output,int n, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void computeMax(Ciphertext &input1,Ciphertext &input2,Ciphertext &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void quickMax(TensorCipher &input,TensorCipher &output,int n, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);

void fakeBootstrap(Ciphertext &input,Ciphertext &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys);
