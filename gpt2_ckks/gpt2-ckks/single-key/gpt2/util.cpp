#include "gpt2_seal.h"

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
	vector<double> vec;
    for(int i=0; i<static_cast<int>(data.size()); i++) vec.emplace_back(data[i]);
    for(int i=data.size(); i<1<<logn; i++) vec.emplace_back(0);      // zero padding

    // vec size = n
    if(vec.size() != static_cast<long unsigned int>(1<<logn)) throw std::out_of_range("the size of vec is not n");

	// encode & encrypt
	Plaintext plain;
	Ciphertext cipher;
	double scale = pow(2.0, logp);
	encoder.encode(vec, scale, plain);
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

void memory_save_rotate(const Ciphertext &cipher_in, Ciphertext &cipher_out, int steps, Evaluator &evaluator, GaloisKeys &gal_keys)
{
	
	long n = cipher_in.poly_modulus_degree() / 2;
	Ciphertext temp = cipher_in;
	steps = (steps+n)%n;	// 0 ~ n-1
	int first_step = 0;
	printf("About to rotate \n");

	if(34<=steps && steps<=55) first_step = 33;
	else if(57<=steps && steps<=61) first_step = 33;
	else first_step = 0;
	if(steps == 0) return;		// no rotation
	if(first_step == 0) evaluator.rotate_vector_inplace(temp, steps, gal_keys);
	else
	{
		evaluator.rotate_vector_inplace(temp, first_step, gal_keys);
		evaluator.rotate_vector_inplace(temp, steps-first_step, gal_keys);
	}

	cipher_out = temp;
//	else scale_evaluator.rotate_vector(cipher_in, steps, gal_keys, cipher_out);
}