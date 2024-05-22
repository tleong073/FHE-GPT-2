#include "approx.h"
#include <cassert>

using namespace std;
using namespace seal;
using namespace minicomp;


void col_matrix_multiplication_seal_print( vector<TensorCipher> &left_inputs, vector<TensorCipher> &right_inputs, vector<TensorCipher> &outputs, vector<double> bias,
	int rows, int cols, Config &config, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys) {
		return;
}

/**
 * @brief Performs a matrix-matrix multiplication on two column packed matrices. 
 *		
 * Performs A x B^T for |A| = m x n and |B| = m x n.
 * NOTE: B is packed like: B || B || .....
 * 
 * @param current the call
 * @param argMap current procedure's arguments
 */
void col_matrix_multiplication_seal( vector<TensorCipher> &left_inputs, vector<TensorCipher> &right_inputs, vector<TensorCipher> &outputs, vector<double> bias,
	int rows, int cols, Config &config, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys) {

	// Initalize Output array to all 0s.
	vector<vector<double>> output_initialize(cols,vector<double>(cols,0.0));
	vector<Ciphertext> output_pre_tensor;

	Ciphertext cipher;
	Plaintext plain;
	Plaintext scaler;
	Ciphertext tmp_cipher;

	double scale;
	parms_id_type parm_id = left_inputs[0].cipher().parms_id();
	double input_scale = left_inputs[0].cipher().scale();

	cout << fixed << setprecision(10);

	for(int i = 0; i<cols; i++) {
		// Initialize ciphertexts
		encoder.encode(output_initialize[i],left_inputs[0].cipher().scale(), plain);
		encryptor.encrypt(plain, cipher);
		
		// Use a "scaler"(encrypted 1) to adjust scale.
		encoder.encode(1,input_scale,scaler);
		encryptor.encrypt(scaler,tmp_cipher);

		// Multiply and rescale to ensure output is init_scale - 1
		evaluator.multiply_inplace(cipher,tmp_cipher);
		evaluator.rescale_to_next_inplace(cipher);

		output_pre_tensor.push_back(cipher);
	}

	cout << "rows and cols: " << rows <<" " <<cols << endl;
	//  For each row, compute corresponding diagonal of product.
	
	for(int i = 0; i< cols; i++) {

		// Multiply and Accumulate diagonal j
		// ri = ri + (aj x bj)
		for(int j = 0; j< rows; j++) {

			//decrypt_and_print_and_max_round(output_pre_tensor[i],decryptor,encoder,1.0,0);
			evaluator.multiply(left_inputs[j].cipher(),right_inputs[j].cipher(),cipher);

			// Relinearize and rescale.
			evaluator.relinearize_inplace(cipher, relin_keys);
			evaluator.rescale_to_next_inplace(cipher);
			parm_id = cipher.parms_id();

			// Adjust output scale and level.
			evaluator.mod_switch_to_inplace(output_pre_tensor[i], parm_id);

			//printf("Scales: output_pre_tensor: %f cipher: %f \n",
				//output_pre_tensor[i].scale(),cipher.scale());

			//printf("Mods: output_pre_tensor: %d cipher: %d \n",
				//output_pre_tensor[i].coeff_modulus_size(),cipher.coeff_modulus_size());

			evaluator.add_inplace_reduced_error(output_pre_tensor[i],cipher);
			//decrypt_and_print_and_max_round(output_pre_tensor[i],decryptor,encoder,1.0,0);
		}
		// Warning: Dark magic at play. We assume the cols are concat together, which allows us to get "circular"
		// rotation even though the vector isn't fully packed.
		for(int j = 0; j<rows;j++) {
			cipher = right_inputs[j].cipher();
			evaluator.rotate_vector_inplace(cipher, 1, gal_keys);
			right_inputs[j].set_ciphertext(cipher);
		}
		//cout << "Past second loop\n" << endl;
	}
	
	TensorCipher tensor;
	
	for(int i = 0; i<cols; i++) {
		tensor = TensorCipher();
		tensor.set_ciphertext(output_pre_tensor[i]);
		outputs.push_back(tensor);
	}
	//cout << "\n\nComplete\n\n" << endl;
    return;
}

/**
 * @brief Performs a cipher-plain multiplication on row-packed matrices. 
 *		
 * Performs A x B for |A| = m x n and |B| = m x n.
 * 
 * 
 * @param current the call
 * @param argMap current procedure's arguments
 */
void row_matrix_multiplication_seal( vector<vector<TensorCipher>> &left_inputs, vector<vector<double>> &weights, vector<TensorCipher> &outputs, vector<double> bias,
	int rows, int cols, Config &config, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys) {
		// Make sure rows match size
		assert(left_inputs[0].size() == weights[0].size());

		Plaintext plain;
		Ciphertext cipher, tmp_cipher;
		vector<Plaintext> encoded_weights;

		TensorCipher t_cipher;

		// Encode weights
		for(int i = 0; i< cols; i++) {
			encoder.encode(weights[i],left_inputs[0][0].cipher().scale(),plain);
			encoded_weights.push_back(plain);
		}

		// Iterate over rows of input ciphers and mult by cipher
		for(int i = 0; i<rows;i++) {
			// Initialize accumulator
			evaluator.multiply_plain(left_inputs[i][0].cipher(),encoded_weights[0],cipher);
			evaluator.rescale_to_next_inplace(cipher);

			for(int j = 1; j< cols;j++) {
				// There is one cipher-mul per column element
				evaluator.multiply_plain(left_inputs[i][j].cipher(),encoded_weights[j],tmp_cipher);
				evaluator.rescale_to_next_inplace(tmp_cipher);
				evaluator.add_inplace_reduced_error(cipher,tmp_cipher);
			}
			
			outputs.push_back(TensorCipher(cipher));
		}
	return;
}

/**
 * @brief Repacks a diagonal packed matrix into a row-wise one for later steps.
 *		
 * Performs A x B for |A| = m x n and |B| = m x n.
 * 
 * 
 * @param current the call
 * @param argMap current procedure's arguments
 */
void diagonal_to_row_matrix_seal( vector<TensorCipher>&inputs,vector<TensorCipher> &outputs, int rows, int cols, Config &config,
	 CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys) {

		Plaintext plain;
		Ciphertext cipher, tmp_cipher;
		vector<Plaintext> encoded_weights;

		TensorCipher t_cipher;
		int start = 0;
		double scale = inputs[0].cipher().scale();

		// Iterate column-wise over diagonals
		for(int i = 0; i< cols; i++) {
			vector<double> mask(cols,0.0);
			mask[i] = 1.0;
			encoder.encode(mask,scale,plain);
			evaluator.multiply_plain(inputs[0].cipher(),plain,cipher);
			evaluator.rescale_to_next_inplace(cipher);
			evaluator.rotate_vector_inplace(cipher,i,gal_keys);

			for(int j = 1; j<rows; j++) {
				// Mask out and rotate into appropriate position.
				evaluator.multiply_plain(inputs[j].cipher(),plain,tmp_cipher);
				evaluator.rescale_to_next_inplace(tmp_cipher);
				evaluator.rotate_vector_inplace(tmp_cipher,-(j-i),gal_keys);
				//memory_save_rotate(tmp_cipher,tmp_cipher,-j,evaluator,gal_keys);

				// Accumulate.
				evaluator.add_inplace(cipher,tmp_cipher);
			}
			outputs.push_back(TensorCipher(cipher));
		}
		printf("Complete \n");
	return;
}