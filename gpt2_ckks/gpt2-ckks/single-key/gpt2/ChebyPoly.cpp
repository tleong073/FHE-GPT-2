#include "approx.h"

/*
class ChebyPoly
{
public:
	// Assume basis consists of even-numbered chebyshev polynomials + 1
	// [T0,T1,T2,T4,T8,....]
	vector<double> coeffs;
	ChebyPoly *q;
	ChebyPoly *r;
	bool isLeaf;
	bool isQ;
	ChebyPoly();
	ChebyPoly(vector<vector<double>> coeffs, ChebyPoly *q, ChebyPoly *r, bool isLeaf, bool isQ);
	void BuildChebyBasis(TensorCipher &input, TensorCipher &chebyBasis, int n);
	void Eval(vector<TensorCipher> chebyBasis, TensorCipher &output);
}
*/

ChebyPoly::ChebyPoly() 
{
	this->_coeffs = vector<double>(1,0);
	this->_q = NULL;
	this->_r = NULL;
	this->_isLeaf = false;
	this->_isQ = false;
}

ChebyPoly::ChebyPoly(vector<vector<double>> coeffs, ChebyPoly *q, ChebyPoly *r, bool isLeaf, bool isQ)
{
	this->_coeffs = coeffs;
	this->_q = q;
	this->_r =r;
	this->_isLeaf = isLeaf;
	this->_isQ = isQ;
}

/**
 * @brief Builds a log chebyshev basis up to N
 *		
 * Uses 3log(N)
 * 
 * 
 * @param input The ciphertext to build the basis over
 * @param chebyBasis The ciphertexts to store the basis in. Assumed empty.
 */
void BuildChebyBasis(CKKSObjects ckks_obs,TensorCipher &input, vector<TensorCipher> &chebyBasis, int n)
{
	// Not defined for negative, and we get 
	assert(n > 2);
	assert(pow(2.0,log(n)) == n);

	size_t slots = input.cipher().poly_modulus_degree() / 2;


	Evaluator *eval;

	Ciphertext cipher;
	Plaintext plain;
	TensorCipher tensor;
	double init_scale = input.cipher().scale();

	// Encode 1's for first polynomial T0
	vector<double> ones(slots,1.0);
	ckks_obs->encoder.encode(ones,init_scale,plain);
	ckks_obs->encrypt(plain,cipher);
	
	tensor = TensorCipher(cipher);
	chebyBasis.push_back(tensor);

	cipher = input.cipher(); 
	// Push back T1-Tn
	for(int i = 0; i<int(log(n));i++) {
		chebyBasis.push_back(TensorCipher(cipher));

		// 2T^2 - 1
		ckks_obs->evaluator->square_inplace(cipher);
		ckks_obs->evaluator->relinearize_inplace(cipher);
		ckks_obs->evaluator->rescale_to_next_inplace(cipher);
		ckks_obs->evaluator->multiply_const_inplace(cipher,2.0);
		ckks_obs->evaluator->rescale_to_next_inplace(cipher);
		ckks_obs->evaluator->add_const_inplace(cipher,-1.0);
	}
}

void evalHelper(CKKSObjects &ckks_obs, ChebyPoly*cheby_poly, vector<TensorCipher> chebyBasis,Ciphertext &output)
{
	printf("Inside eval helper");
	Ciphertext tmp_cipher;
	// Leaf case is to simply eval polynomial using basis.
	if(cheby_poly->_isLeaf) {
		printf("Leaf case");
		bool init = false;
		for(int i = 0; i<cheby_poly->_coeffs.size();i++) {
			if(cheby_poly->_coeffs[i] == 0)
				continue;
			// A bit sloppy, but less messy than initializing a zero text and accumulating.
			if(!init){
				ckks_obs.evaluator->multiply_const(chebyBasis[i],cheby_poly->_coeffs[i],output);
				ckks_obs.evaluator->rescale_to_next_inplace(output);
				init= true;
			}else {
				ckks_obs.evaluator->multiply_const(chebyBasis[i],cheby_poly->_coeffs[i],output);
				ckks_obs.evaluator->rescale_to_inplace(output);
				// Rescale if necessary
				if(tmp_cipher.scale() < acc_cipher.scale()) {
					ckks_obs.evaluator->rescale_to_inplace(output,tmp_cipher.parms_id());
				}
				ckks_obs.evaluator->add_inplace(acc_cipher,output);
			}
		}
		return;
	}

	// Recursive Case
}
/**
 * @brief Evaluates the given chebyshev polynomial
 *		
 * 
 * 
 * @param input The ciphertext to build the basis over
 * @param chebyBasis The ciphertexts to store the basis in. Assumed empty.
 */
void Eval(vector<TensorCipher> &chebyBasis, TensorCipher &output) {

	Ciphertext cipher,out_cipher;
	return;
}