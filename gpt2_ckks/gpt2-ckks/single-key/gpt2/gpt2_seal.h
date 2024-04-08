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

void col_matrix_multiplication_seal( vector<TensorCipher> &left_inputs, vector<TensorCipher> &right_inputs, vector<TensorCipher> &outputs, vector<double> bias,
	int rows, int cols, Config &config, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);

void memory_save_rotate(const Ciphertext &cipher_in, Ciphertext &cipher_out, int steps, Evaluator &evaluator, GaloisKeys &gal_keys);

void decrypt_and_print(const Ciphertext &cipher, Decryptor &decryptor, CKKSEncoder &encoder,int n);

void row_matrix_multiplication_seal( vector<vector<TensorCipher>> &left_inputs, vector<vector<double>> &weights, vector<TensorCipher> &outputs, vector<double> bias,
	int rows, int cols, Config &config, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys);