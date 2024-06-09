#include "util.h"
#include <math.h>

using namespace std;
using namespace seal;

TensorCipher::TensorCipher()
{
    k_=0;
    h_=0;
    w_=0;
	c_=0;
	t_=0;
    p_=0;
}
TensorCipher::TensorCipher(Ciphertext cipher)
{
    k_=0;
    h_=0;
    w_=0;
	c_=0;
	t_=0;
    p_=0;
	this->cipher_ = cipher;
}
TensorCipher::TensorCipher(int logn, int k, int h, int w, int c, int t, int p, vector<double> data, Encryptor &encryptor, CKKSEncoder &encoder, int logp)
{
    if(k != 1) throw std::invalid_argument("supported k is only 1 right now");
    
	// 1 <= logn <= 16
    if(logn < 1 || logn > 16) throw std::out_of_range("the value of logn is out of range");
    if(data.size() > static_cast<long unsigned int>(1<<logn)) throw std::out_of_range("the size of data is larger than n");

    this->k_ = k;
    this->h_ = h;
	this->w_ = w;
	this->c_ = c;
    this->t_ = t;
	this->p_ = p;
	this->logn_ = logn;

	// generate vector that contains data
	vector<double> v;
    for(int i=0; i<static_cast<int>(data.size()); i++) v.emplace_back(data[i]);
    for(int i=data.size(); i<1<<logn; i++) v.emplace_back(0);      // zero padding

    // vec size = n
    if(v.size() != static_cast<long unsigned int>(1<<logn)) throw std::out_of_range("the size of vec is not n");

	// encode & encrypt
	Plaintext plain;
	Ciphertext cipher;
	double scale = pow(2.0, logp);
	encoder.encode(v, scale, plain);
	encryptor.encrypt(plain, cipher);
	this->set_ciphertext(cipher);

}
TensorCipher::TensorCipher(int logn, int k, int h, int w, int c, int t, int p, Ciphertext cipher)
{
    this->k_ = k;
    this->h_ = h;
	this->w_ = w;
	this->c_ = c;
    this->t_ = t;
	this->p_ = p;
	this->logn_ = logn;
	this->cipher_ = cipher;
}
int TensorCipher::k() const
{
	return k_;
}
int TensorCipher::h() const
{
	return h_;
}
int TensorCipher::w() const
{
	return w_;
}
int TensorCipher::c() const
{
	return c_;
}
int TensorCipher::t() const
{
	return t_;
}
int TensorCipher::p() const
{
	return p_;
}
int TensorCipher::logn() const
{
	return logn_;
}
Ciphertext TensorCipher::cipher() const
{
	return cipher_;
}
void TensorCipher::set_ciphertext(Ciphertext cipher)
{
	cipher_ = cipher;
}
void TensorCipher::print_parms()
{
	cout << "k: " << k_ << endl;
    cout << "h: " << h_ << endl;
    cout << "w: " << w_ << endl;
	cout << "c: " << c_ << endl;
	cout << "t: " << t_ << endl;
	cout << "p: " << p_ << endl;
}

Config::Config() {
	this->boundary_K = 25;
	this->boot_deg = 59;
	this->scale_factor = 2;
	this->inverse_deg = 1; 
	this->logN = 16;
	this->loge = 10; 
	this->logn = 15;		// full slots
	this->logn_1 = 14;	// sparse slots
	this->logn_2 = 13;
	this->logn_3 = 12;
	this->logp = 46;
	this->logq = 51;
	this->log_special_prime = 51;
	this->log_integer_part = this->logq - this->logp - this->loge + 5;
	this->remaining_level = 16;
	this->boot_level = 14;
	this->total_level = this->remaining_level + this->boot_level;
}

Config::Config(
		long boundary_K,
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
		int total_level)
{
	this->boundary_K = boundary_K;
	this->boot_deg = boot_deg;
	this->scale_factor = scale_factor;
	this->inverse_deg = inverse_deg; 
	this->logN = logN;
	this->loge = loge; 
	this->logn = logn;
	this->logn_1 = logn_1;
	this->logn_2 = logn_2;
	this->logn_3 = logn_3;
	this->logp = logp;
	this->logq = logq;
	this->log_special_prime = log_special_prime;
	this->log_integer_part = logq - logp - loge + 5;
	this->remaining_level = remaining_level; // Calculation required
	this->boot_level = boot_level; // 
	this->total_level = remaining_level + boot_level;
}

/*
class CKKSObjects
{
public:
	CKKSEncoder *encoder;
	Encryptor *Encryptor;
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
}
*/

CKKSObjects::CKKSObjects() 
{
	this->encoder = NULL;
	this->decryptor = NULL;
	this->evaluator = NULL;
	this->encryptor = NULL;
	this->gal_keys = NULL;
	this->relin_keys = NULL;
}


//CITE: copied from FHE-CNN-CKKS
void rotate_inplace(Ciphertext &cipher_in, int steps, Evaluator &evaluator, GaloisKeys &gal_keys)
{	
	int n = cipher_in.poly_modulus_degree() / 2;
	int rot_amt = steps;
	if(steps == 0) return;		// no rotation
	
	
	if(steps < 0){
		rot_amt = 32768+steps;
	} 
	printf("Rot amt: %d\n",rot_amt);
	if(rot_amt == 16384){
		evaluator.rotate_vector_inplace(cipher_in,rot_amt-2048,gal_keys);
		evaluator.rotate_vector_inplace(cipher_in,2048,gal_keys);
	} else {
		evaluator.rotate_vector_inplace(cipher_in,rot_amt,gal_keys);
	}
	
//	else scale_evaluator.rotate_vector(cipher_in, steps, gal_keys, cipher_out);
}

void rotate_vec(const Ciphertext &cipher_in, Ciphertext & cipher_out, int steps, Evaluator &evaluator, GaloisKeys &gal_keys)
{	
	int n = cipher_in.poly_modulus_degree() / 2;
	int rot_amt = steps;
	if(steps == 0) return;		// no rotation
	
	
	if(steps < 0){
		rot_amt = 32768+steps;
	} 

	printf("Rot amt: %d\n",rot_amt);
	if(rot_amt == 16384){
		evaluator.rotate_vector(cipher_in,rot_amt-2048,gal_keys,cipher_out);
		evaluator.rotate_vector_inplace(cipher_out,2048,gal_keys);
	} else {
		evaluator.rotate_vector(cipher_in,rot_amt,gal_keys,cipher_out);
	}
	
//	else scale_evaluator.rotate_vector(cipher_in, steps, gal_keys, cipher_out);
}
/*
def round_to_2(x):
    return math.pow(2,math.ceil(math.log(x,2)))
*/
int round_to_2(double x) {
	return pow(2.0,ceil(log2(x)));
}

/**
 * @brief Bootstrap replacement. 
 * 
 */
void fakeBootstrap(Ciphertext &input, Ciphertext &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys) {
	Plaintext plain;
	vector<double> res;

	decryptor.decrypt(input,plain);
	encoder.decode(plain, res);
	encoder.encode(res, pow(2,51),plain);
	encryptor.encrypt(plain, output);
}

void init_output(int num_ciphers,vector<Ciphertext> &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys) {		
						
	vector<double> x(1,0.0);
	Plaintext plain;
	Ciphertext cipher;

	for(int i = 0; i<num_ciphers;i++){
		encoder.encode(x,ENCODE_SCALE,plain);
		encryptor.encrypt(plain,cipher);
		output.push_back(cipher);
	}
}

// Masks out length elements starting from start
void mask_out(Ciphertext &cipher,Ciphertext &out, int start,int length,CKKSEncoder &encoder,Evaluator &evaluator,RelinKeys &relin_keys) {
    vector<double> x(32768,0.0);
    Plaintext plain;

    fill(x.begin()+start,x.begin()+start+length,1.0);
	printf("mask_out start: %d length: %d \n",start,length);

    encoder.encode(x, ENCODE_SCALE,plain);
	
    evaluator.multiply_plain(cipher,plain,out);
	printf("mask_out mul done : %d\n",out.is_transparent());
    evaluator.relinearize_inplace(out,relin_keys);
    evaluator.rescale_to_next_inplace(out);
}

// Row packs a matrix into a ciphertext
void pack_plain_row(vector<vector<double>>& v,int rows,int row_size,vector<vector<double>> &out) {
	int rounded_row_size = round_to_2(row_size);
	int pos=0;
	
	for(int i = 0;i<rows;i++){
		for(int j = 0; j<row_size;j++) {
			pos = i*rounded_row_size*2+j;
			//printf("%d %d %d %d %d %f\n",pos,pos/32768,pos%32768,i,j,vec[i][j]);
			out[pos / 32768][pos % 32768] = v[i][j];
		}
	}
	return;
}