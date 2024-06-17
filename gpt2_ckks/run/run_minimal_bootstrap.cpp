#include "seal/seal.h"
#include <complex>
#include <iostream>
#include <fstream>
#include <cmath>
#include <NTL/RR.h>
#include "Polynomial.h"
#include "ModularReducer.h"
#include "Bootstrapper.h"
// #include "ScaleInvEvaluator.h"
#include <random>
#include <chrono>
#include "util.h"

using namespace std;
using namespace NTL;
using namespace seal;
using namespace chrono;

int main() {

	long boundary_K = 25;
	long deg = 59;
    long scale_factor = 2;
    long inverse_deg = 1; 

	long logN = 16;
	long loge = 10;

	long logn = logN-1;
	long logn_2 = 13;
	long logn_3 = 12;
	long sparse_slots = (1 << logn);

	int logp = 46;
	int logq = 51;
	int log_special_prime = 51;

    int log_integer_part = logq - logp - loge + 5;

	// int remaining_level = 14; // Calculation required
	int remaining_level = 21; // Calculation required
	int boot_level = 14; // greater than: subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2 
	int total_level = remaining_level + boot_level;

	Plaintext pt;
	Ciphertext ct,rct;


    // Macro to avoid typing shit in for tests
    INIT();

	// vec v(5,11.0);
	// encoder.encode(v,scale,pt);
	// encryptor.encrypt(pt,ct);

	// vector<complex<double>> tmpvec(32768, 0);
	// for (int i = 0; i < 10; i++) {
	// 	for (int j = 0; j < 10; j++) {
	// 		tmpvec[i * (2 * 10) + j] = -0.5;
	// 		tmpvec[i * (2 * 10) + j + 10] = -0.5; 
	// 	}
	// }

	// evaluator.multiply_vector_reduced_error(ct,tmpvec,rct);

	// evaluator.complex_conjugate_inplace(rct,gal_keys);
	
	// return 0;

	size_t slot_count = encoder.slot_count();
	
	Bootstrapper bootstrapper(loge, logn, logN - 1, total_level, scale, boundary_K, deg, scale_factor, inverse_deg, context, keygen, encoder, encryptor, decryptor, evaluator, relin_keys, gal_keys);

    cout << "Generating Optimal Minimax Polynomials..." << endl;
    bootstrapper.prepare_mod_polynomial();
    cout << "Adding Bootstrapping Keys..." << endl;
    //bootstrapper.addBootKeys_3_other_slots(gal_keys, slot_vec);
	bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);
	keygen.create_galois_keys(gal_steps_vector, gal_keys);

    bootstrapper.slot_vec.push_back(logn);

	cout << "Generating Linear Transformation Coefficients..." << endl;
	bootstrapper.generate_LT_coefficient_3();

	double tot_err = 0, mean_err;
	size_t iterations = 2;

	vec input(slot_count,2.0);
	vec before(slot_count);
	vec after(slot_count);
	
	Plaintext plain;
	Ciphertext cipher;
	
	for (size_t _ = 0; _ < iterations; _++)
	{	
        sparse_slots = (1 << logn);

		cout << _ << "-th iteration : sparse_slots = " << sparse_slots << endl;
	
		encoder.encode(input, scale, plain);
		encryptor.encrypt(plain, cipher);
		
		int reduce = cipher.coeff_modulus_size()-1;
		for (int i = 0; i < reduce; i++) {
			evaluator.mod_switch_to_next_inplace(cipher);
		}

		Ciphertext rtn;

		decryptor.decrypt(cipher, plain);
		encoder.decode(plain, before);
		
        auto start = system_clock::now();
		cout << " Old level: " << cipher.coeff_modulus_size() << endl;
		bootstrapper.bootstrap_real_3(rtn, cipher);
		cout << " New level: " << rtn.coeff_modulus_size() << endl;

        duration<double> sec = system_clock::now() - start;
        cout << "bootstrapping time : " << sec.count() << "s" << endl;

		decryptor.decrypt(rtn, plain);
		// encoder.decode(plain, after, sparse_slots);
		encoder.decode(plain, after);
	
		mean_err = 0;
		for (long i = 0; i < sparse_slots; i++)
		{
			//cout << i << "m: " << recover_vefore(before[i].real(), boundary_K) << "\td: " << after[i].real()<< endl;
		
            if (i < 10) cout << i << " " << before[i] << " " << after[i] << endl;

			mean_err += abs(before[i] - after[i]);
		}
		mean_err /= 2*sparse_slots;
		cout << "Absolute mean of error: " << mean_err << endl;
		
		tot_err += mean_err;

		cipher = rtn;

	}
	tot_err /= iterations;
	cout << " mean error: " << tot_err << endl;
	
    

	return 0;
}
