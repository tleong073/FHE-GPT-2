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