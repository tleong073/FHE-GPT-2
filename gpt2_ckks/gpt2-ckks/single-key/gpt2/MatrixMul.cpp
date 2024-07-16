#include "approx.h"
#include "util.h"
#include <cassert>
#include <chrono>

using namespace std;
using namespace seal;
using namespace minicomp;
using namespace chrono;

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
 * Performs A x B.T for |A| = m x n and |B| = m x n.
 * Assumes Matrices are packed into fold format
 * 
 * @param current the call
 * @param argMap current procedure's arguments
 */
void row_matrix_multiplication_seal( vector<Ciphertext> &left_inputs, vector<Ciphertext> &weights,Ciphertext bias, vector<Ciphertext> &outputs,
	int A_rows, int A_cols,int W_rows,int W_cols, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys) {

		// Row and column dimension computations
		int W_rows_rounded = round_to_2(W_rows);
		int W_cols_rounded = round_to_2(W_cols);
		printf("W_cols: %d W_rows %d\n",W_cols_rounded,W_rows_rounded);

		int chunk_size = W_rows_rounded * 2;
		int num_chunks = 32768 / chunk_size;
		int out_chunk_size = W_cols_rounded*2;
		int row,col,cipher_idx,cipher_chunk,chunk_offset;
		int desired_location, shift_amt;

		Plaintext plain;
		Ciphertext cipher, rolled,folded,res,res1,masked_out,tmp_cipher;
		vector<Plaintext> encoded_weights;

		TensorCipher t_cipher,tmp_tcipher;

		for(int i = 0; i<left_inputs.size();i++){
			for(int j = 0; j<weights.size();j++) {
				for(int rots = 0; rots < num_chunks;rots++){
					printf("Before rotate1 : %d %d %d %d\n",i,j,rots,rots*chunk_size);
					// Obtain Hademard Product results
					//evaluator.rotate_vector(weights[j],rots*chunk_size,gal_keys,rolled);
					//decrypt_and_print_and_max_round(weights[j],decryptor,encoder,1.0,0);
					evaluator.rotate_vector(weights[j],rots*chunk_size,gal_keys,rolled);
					//decrypt_and_print_and_max_round(weights[j],decryptor,encoder,1.0,0);
					evaluator.multiply_reduced_error(left_inputs[i],rolled,relin_keys,res1);
					evaluator.rescale_to_next_inplace(res1);

					printf("Formatting for fold\n");
					// Format for fold and perform fold
					//evaluator.rotate_vector(res1,W_rows_rounded,gal_keys,rolled);
					evaluator.rotate_vector(res1,32768-W_rows_rounded,gal_keys,rolled);
					evaluator.add_inplace_reduced_error(res1,rolled);

					printf("Summing\n");
					// QuickSum across all ciphertexts
					//decrypt_and_print_and_max_round(res1,decryptor,encoder,1.0,0);
					quickSum(res1,folded,W_rows_rounded,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
					//decrypt_and_print_and_max_round(folded,decryptor,encoder,1.0,0);

					for(int pos = 0;pos<num_chunks;++pos){
						row = i * num_chunks + pos;
						col = j * num_chunks + ((rots+pos)  % num_chunks);

						// Mask out real chunk
						mask_out(folded,masked_out,pos*chunk_size,1,encoder,evaluator,relin_keys);

						cipher_idx = (row*out_chunk_size) / 32768;
						cipher_chunk = ((row*out_chunk_size) % 32768) / out_chunk_size;
						chunk_offset = col;

						desired_location = cipher_chunk*out_chunk_size + chunk_offset;
						shift_amt = desired_location - pos*chunk_size;

						evaluator.rotate_vector_inplace(masked_out,-shift_amt,gal_keys);
						evaluator.add_inplace_reduced_error(outputs[cipher_idx],masked_out);
					}
				}
			}
		}

	printf("Adding bias \n");
	for(int i =0 ;i<outputs.size();++i){
		evaluator.add_inplace_reduced_error(outputs[i],bias);
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

/**
 * @brief Performs projection into attention layer 
 *		
 * Performs A x B.T for |A| = m x n and |B| = m x n.
 * Assumes Matrices are packed into fold format
 * 
 * @param current the call
 * @param argMap current procedure's arguments
 */
void attn_proj_row_seal( vector<Ciphertext> &left_inputs, vector<Ciphertext> &weights,Ciphertext bias, vector<Ciphertext> &outputs,
	int A_rows, int A_cols,int W_rows,int W_cols,KeyGenerator &keygen, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys)
{

		// Row and column dimension computations

		Ciphertext cipher, rolled,folded,res,res1,masked_out,tmp_cipher;
		vector<Plaintext> encoded_weights;

		vvec masks;
		vec v(32768,0.0);
		int k;
		for(k=0;k<16;k++){
			v = vec(32768,0.0);
			v[k*2048] = 1.0;
			masks.push_back(v);
		}

		vc cipher_pool(32);
		vc working_pool(32*16);

		duration<double> sec; 
		omp_lock_t pool_lock;
		omp_init_lock(&pool_lock);
		vector<omp_lock_t> head_locks(12);
		vector<omp_lock_t> thread_locks(32);
		// Set up locks
		for(int i=0;i<12; i++){
 			omp_init_lock(&head_locks[i]);
		}
		for(int i=0;i<32; i++){
			omp_init_lock(&thread_locks[i]);
		}



		// Cannot assume size, since KV-caching requires two different sizes
		#pragma omp parallel for num_threads(THREAD_NUM) collapse(2)
		for(int i = 0; i<left_inputs.size();i++){
			for(int j = 0; j<weights.size();j++){
				auto time = system_clock::now();
				int id = omp_get_thread_num();
				//printf("I am thread: %d\n",id);

				for(int k = 0; k<16;k++){
					//printf("I am a looping thread: %d %d %d\n",id,k,id*16+k);
					evaluator.multiply_vector_reduced_error(weights[j],masks[0],working_pool[id*16+k]);
					evaluator.rescale_to_next_inplace(working_pool[id*16+k]);
					
					evaluator.rotate_vector_inplace(working_pool[id*16+k],-1024,gal_keys);
					quickSum(working_pool[id*16+k],working_pool[id*16+k],1024,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

				}
				for(int rots = 0; rots<16;rots++){
					for(int pos = 0; pos<16; pos++) {
						int abs_pos,head_col,head,desired_location,shift_amt;

						abs_pos = (i * 16 + pos)*W_cols+(j * 16 + ((rots + pos) % 16));

						head_col = abs_pos % 64;
						head = (abs_pos / 64) % 12;

						//printf("HERE I AM THREAD: %d\n",id);
						//mask_out(cipher,l_cipher,pos*2048,1,encoder,evaluator,relin_keys);
						evaluator.multiply_vector_reduced_error(working_pool[id*16+rots],masks[pos],cipher_pool[id]);
						evaluator.rescale_to_next_inplace(cipher_pool[id]);
						
						//decrypt_and_print_and_max_round(masked_out,decryptor,encoder,1.0,0);
						desired_location = (i * 16 + pos)*A_rows+head_col;

						shift_amt = desired_location - pos*2048;
						
						// printf("Shift amt: %d %d %d\n",i,j,shift_amt);
						
						// A bit jank, but we with our limited memory, we cant store all the Galois Keys for now.
						//surefire_rotate(masked_out,shift_amt, keygen, evaluator);
						// !!!!!!!! TAKE OUT WHEN DOING CORRECTNESS. TEMPORARY MEMORY SAVING FEATURE !!!!!!!
						evaluator.rotate_vector_inplace(cipher_pool[id],0,gal_keys);

						//decrypt_and_print_and_max_round(pool2[rots*16+pos],decryptor,encoder,1.0,0);
						//printf("desired_location: %d %d %d %d\n",desired_location,shift_amt,pos,outputs[head].is_transparent());
						omp_set_lock(&head_locks[head]);
						evaluator.add_inplace_reduced_error(outputs[head],cipher_pool[id]);
						omp_unset_lock(&head_locks[head]);
				}
			}
			

			duration<double> sec = system_clock::now() - time;
    		cout << "attn_proj time : " << sec.count() << "s" << endl;
		}
	}

	// Tear down locks
	omp_destroy_lock(&pool_lock);
	for(int i=0;i<12; i++){
		omp_destroy_lock(&head_locks[i]);
	}
	for(int i=0;i<32; i++){
		omp_destroy_lock(&thread_locks[i]);
	}
	
	for(int i = 0; i<outputs.size();i++)
		evaluator.add_inplace_reduced_error(outputs[i],bias);
	return;
}

/**
 * @brief Performs projection into attention layer with output packed into cols
 *		
 * Performs A x B.T for |A| = m x n and |B| = m x n.
 * Assumes Matrices are packed into fold format
 * 
 * @param current the call
 * @param argMap current procedure's arguments
 */
void attn_proj_col_seal( vector<Ciphertext> &left_inputs, vector<Ciphertext> &weights,Ciphertext bias, vector<Ciphertext> &outputs,
	int A_rows, int A_cols,int W_rows,int W_cols,KeyGenerator &keygen, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys) {


	// Row and column dimension computations

		Ciphertext cipher, rolled,folded,res,res1,masked_out,tmp_cipher;
		vector<Plaintext> encoded_weights;

		vvec masks;
		vec v(32768,0.0);
		int k;
		for(k=0;k<16;k++){
			v = vec(32768,0.0);
			v[k*2048] = 1.0;
			masks.push_back(v);
		}

		vc cipher_pool(32);
		vc working_pool(32*16);

		duration<double> sec; 
		omp_lock_t pool_lock;
		omp_init_lock(&pool_lock);
		vector<omp_lock_t> head_locks(12);
		vector<omp_lock_t> thread_locks(32);
		// Set up locks
		for(int i=0;i<12; i++){
 			omp_init_lock(&head_locks[i]);
		}
		for(int i=0;i<32; i++){
			omp_init_lock(&thread_locks[i]);
		}



		// Cannot assume size, since KV-caching requires two different sizes
		#pragma omp parallel for num_threads(THREAD_NUM) collapse(2)
		for(int i = 0; i<left_inputs.size();i++){
			for(int j = 0; j<weights.size();j++){
				auto time = system_clock::now();
				int id = omp_get_thread_num();
				//printf("I am thread: %d\n",id);

				for(int k = 0; k<16;k++){
					//printf("I am a looping thread: %d %d %d\n",id,k,id*16+k);
					evaluator.multiply_vector_reduced_error(weights[j],masks[0],working_pool[id*16+k]);
					evaluator.rescale_to_next_inplace(working_pool[id*16+k]);
					
					evaluator.rotate_vector_inplace(working_pool[id*16+k],-1024,gal_keys);
					quickSum(working_pool[id*16+k],working_pool[id*16+k],1024,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
				}
				for(int rots = 0; rots<16;rots++){
					for(int pos = 0; pos<16; pos++) {
						int row,col,abs_pos,head_row,head_col,head,desired_location,shift_amt;

						row = i * 16 + pos;
						col = j * 16 + ((rots + pos) % 16);

						abs_pos = row*768+col;

						head_row = abs_pos / 64;
						head_col = abs_pos % 64;
						head = head_row % 12;

						//printf("HERE I AM THREAD: %d\n",id);
						//mask_out(cipher,l_cipher,pos*2048,1,encoder,evaluator,relin_keys);
						evaluator.multiply_vector_reduced_error(working_pool[id*16+rots],masks[pos],cipher_pool[id]);
						evaluator.rescale_to_next_inplace(cipher_pool[id]);
						
						//decrypt_and_print_and_max_round(masked_out,decryptor,encoder,1.0,0);
						desired_location = (i * 16 + pos)*A_rows+head_col;

						shift_amt = desired_location - pos*2048;
						
						// printf("Shift amt: %d %d %d\n",i,j,shift_amt);
						
						// A bit jank, but we with our limited memory, we cant store all the Galois Keys for now.
						//surefire_rotate(masked_out,shift_amt, keygen, evaluator);
						// !!!!!!!! TAKE OUT WHEN DOING CORRECTNESS. TEMPORARY MEMORY SAVING FEATURE !!!!!!!
						evaluator.rotate_vector_inplace(cipher_pool[id],0,gal_keys);

						//decrypt_and_print_and_max_round(pool2[rots*16+pos],decryptor,encoder,1.0,0);
						//printf("desired_location: %d %d %d %d\n",desired_location,shift_amt,pos,outputs[head].is_transparent());
						omp_set_lock(&head_locks[head]);
						evaluator.add_inplace_reduced_error(outputs[head],cipher_pool[id]);
						omp_unset_lock(&head_locks[head]);
				}
			}
			

			duration<double> sec = system_clock::now() - time;
    		cout << "attn_proj time : " << sec.count() << "s" << endl;
		}
	}

	// Tear down locks
	omp_destroy_lock(&pool_lock);
	for(int i=0;i<12; i++){
		omp_destroy_lock(&head_locks[i]);
	}
	for(int i=0;i<32; i++){
		omp_destroy_lock(&thread_locks[i]);
	}
	
	for(int i = 0; i<outputs.size();i++)
		evaluator.add_inplace_reduced_error(outputs[i],bias);
	return;
	
}

/**
 * @brief Performs projection into attention layer with output packed into cols
 *		
 * Performs A x B.T for |A| = m x n and |B| = m x n.
 * Assumes Matrices are packed into fold format
 * 
 * @param current the call
 * @param argMap current procedure's arguments
 */
void qk_matmul( vector<Ciphertext> &Q, vector<Ciphertext> &K, vector<Ciphertext> &outputs,
	int A_rows, int A_cols,int W_rows,int W_cols,KeyGenerator &keygen, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys) {

		// Row and column dimension computations
		int row,col,abs_pos,head_row,head_col,head,desired_location,shift_amt;
		Ciphertext cipher, rolled,folded,res,res1,masked_out,tmp_cipher;
		vector<Plaintext> encoded_weights;

		for(int i = 0; i<Q.size();i++){
			// Duplicate right so that left rots eventually contain all products
			evaluator.rotate_vector(K[i],16384,gal_keys,rolled);
			evaluator.add_inplace_reduced_error(K[i],rolled);

			#pragma omp parallel for num_threads(THREAD_NUM)
			for(int rots=0;rots<128;rots++){
				// mul, roll and fold to perform dot product between A's ith row and K's (i+rot) col
				evaluator.multiply_reduced_error(Q[i],K[i],relin_keys,cipher);
				evaluator.rescale_to_next_inplace(cipher);

				evaluator.rotate_vector(cipher,32768-64,gal_keys,rolled);
				quickSum(rolled,folded,64,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
				for(int pos=0;pos<128;pos++){
					row = i * 128 + pos;
					col = i * 128 + ((rots + pos) % 128);

					abs_pos = row*128 + col;
					head_col = abs_pos % 128;

					mask_out(folded,masked_out,pos*128,1,encoder,evaluator,relin_keys);

					desired_location = row*256 + head_col;
                
					shift_amt = desired_location - pos*128;

					// A bit jank, but we with our limited memory, we cant store all the Galois Keys for now.
					surefire_rotate(masked_out,shift_amt, keygen, evaluator);

					evaluator.add_inplace_reduced_error(outputs[i],masked_out);
				}

			}
		}

	return;
}

/**
 * @brief Performs projection into attention layer with output packed into cols
 *		
 * Performs A x B.T for |A| = m x n and |B| = m x n.
 * Assumes Matrices are packed into fold format
 * 
 * @param current the call
 * @param argMap current procedure's arguments
 */
void sv_matmul( vector<Ciphertext> &S, vector<Ciphertext> &V, vector<Ciphertext> &outputs,
	int A_rows, int A_cols,int W_rows,int W_cols,KeyGenerator &keygen, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys) {

		// Row and column dimension computations

		int row,col,abs_pos,head_row,head_col,head,chunk_pos,chunk_offset,cipher_idx,desired_location,shift_amt;
		Ciphertext cipher, rolled,folded,res,res1,masked_out,tmp_cipher;
		vector<Plaintext> encoded_weights;

		int i,rots,pos;
		for(i = 0; i<S.size();i++){
			for(rots=0;rots<64;rots++){
				evaluator.rotate_vector(V[i],32768-16384,gal_keys,cipher);
				evaluator.rotate_vector_inplace(cipher,rots*256,gal_keys);
				// mul, roll and fold to perform dot product between S's ith row and V's (i+rot) col
				evaluator.multiply_inplace_reduced_error(cipher,S[i],relin_keys);
				evaluator.rescale_to_next_inplace(cipher);

				evaluator.rotate_vector(cipher,32768-128,gal_keys,rolled);
				quickSum(rolled,folded,128,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
				for(pos=0;pos<128;pos++){
					row = pos;
					col = ((rots + pos) % 64);

					
					// 1. Compute which chunk of size 768 to insert
					chunk_pos = row;
					// 2. Compute where in chunk to place element
					chunk_offset = (i * 64) + col;
					// 3. Compute which ciphertext the element belongs in
					cipher_idx = chunk_pos; // 16
					//4. Compute desired_location within cipher
					desired_location = (chunk_pos % 16)*2048 + chunk_offset;
					

					mask_out(folded,masked_out,pos*256,1,encoder,evaluator,relin_keys);
                
					shift_amt = desired_location - pos*256;

					// A bit jank, but we with our limited memory, we cant store all the Galois Keys for now.
					surefire_rotate(masked_out,shift_amt, keygen, evaluator);

					evaluator.add_inplace_reduced_error(outputs[cipher_idx],masked_out);
				}

			}
		}

	return;
}

void cipher_plain_128_128( Ciphertext &left_input, unordered_map<string,vector<double>> &weights,Ciphertext bias, vector<Ciphertext> &outputs,
	int A_rows, int A_cols,int W_rows,int W_cols,KeyGenerator &keygen, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys)
{	
	int rots;
	vector<Ciphertext> prod0_left(32),prod0_right(32),prod1_left(32),prod1_right(32),tmp,l_tmp(32),r_tmp(32),out0(32),out1(32);

	#pragma omp parallel for num_threads(THREAD_NUM)
	for(rots=0;rots<128;rots++){
		auto start = system_clock::now();
		int id = omp_get_thread_num();
		// TODO: Replace with proper key once weight infrastrcture is set up
		evaluator.multiply_vector_reduced_error(left_input,weights["test"],prod0_left[id]);
		evaluator.rescale_to_next_inplace(prod0_left[id]);

		evaluator.multiply_vector_reduced_error(left_input,weights["test"],prod0_right[id]);
		evaluator.rescale_to_next_inplace(prod0_right[id]);

		evaluator.multiply_vector_reduced_error(left_input,weights["test"],prod1_left[id]);
		evaluator.rescale_to_next_inplace(prod1_left[id]);

		evaluator.multiply_vector_reduced_error(left_input,weights["test"],prod1_right[id]);
		evaluator.rescale_to_next_inplace(prod1_right[id]);

		//Fold left and right
		quickSum(prod0_left[id],prod0_left[id],128,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
		quickSum(prod0_right[id],prod0_right[id],128,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
		//quickSum(prod1_left[id],prod1_left[id],64,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
		//quickSum(prod1_right[id],prod1_right[id],64,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

		// Accumulate
		evaluator.multiply_vector_reduced_error(prod0_left[id],weights["test"],l_tmp[id]);
		evaluator.multiply_vector_reduced_error(prod0_right[id],weights["test"],r_tmp[id]);
		evaluator.add(l_tmp[id],r_tmp[id],out0[id]);
		evaluator.rescale_to_next_inplace(out0[id]);

		//evaluator.multiply_vector_reduced_error(prod0_left[id],weights["test"],l_tmp[id]);
		//evaluator.multiply_vector_reduced_error(prod0_right[id],weights["test"],r_tmp[id]);
		//evaluator.add(l_tmp[id],r_tmp[id],out1[id]);
		//evaluator.rescale_to_next_inplace(out1[id]);
		duration<double> sec = system_clock::now() - start;
		cout << "cipher_plain_128 : " << sec.count() << "s" << endl;
	}
}

void batch_matmul( vector<Ciphertext> &left_inputs, unordered_map<string,vector<double>> &weights,Ciphertext bias, vector<Ciphertext> &outputs,
	int A_rows, int A_cols,int W_rows,int W_cols,KeyGenerator &keygen, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys) {
		int i,j;
		vc cipher(128);
			
		for(i=0;i<64;i++){
			auto start = system_clock::now();
			#pragma omp parallel for num_threads(THREAD_NUM)
			for(j=0;j<128;j++){
				evaluator.multiply_vector_reduced_error(left_inputs[j],weights["test"],cipher[j]);
				evaluator.rescale_to_next_inplace(cipher[j]);
			}
			evaluator.add_many(cipher,outputs[i]);

			duration<double> sec = system_clock::now() - start;
			cout << "batch mul time : " << sec.count() << "s" << endl;
		}
	
	return;
}

void qk_matmul_col( vector<Ciphertext> &left_input,vector<Ciphertext> &right_input, unordered_map<string,vector<double>> &weights,Ciphertext bias, vector<Ciphertext> &outputs,
	int A_rows, int A_cols,int W_rows,int W_cols,KeyGenerator &keygen, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys) {
		int i,j;
		vc cipher(64);
			
		for(i=0;i<128;i++){
			auto start = system_clock::now();

			#pragma omp parallel for num_threads(THREAD_NUM)
			for(j=0;j<64;j++){
				evaluator.multiply_reduced_error(left_input[j],right_input[j],relin_keys,cipher[j]);
				evaluator.rescale_to_next_inplace(cipher[j]);
			}

			evaluator.add_many(cipher,outputs[i]);

			#pragma omp parallel for num_threads(THREAD_NUM)
			for(j=0;j<64;j++){
				evaluator.rotate_vector_inplace(right_input[j],1,gal_keys);
			}

			duration<double> sec = system_clock::now() - start;
			cout << "batch mul time : " << sec.count() << "s" << endl;
		}
	
	return;
}

/*def sv_matmul(S,V):
    print(S.shape)
    assert S.shape == (12,32768) 
    assert V.shape == (12,32768)

    # Output is (8,32768) matrix
    output = np.zeros((8,32768))
    for i in range(12):
        for rots in range(64):

            # Need to create duplicate format for rotates to work
            # |A|B|C|...|A|B|C|
            dup = V[i] + np.roll(V[i],16384)

            rolled = np.roll(dup,-(rots*256))
            res1 = S[i] * rolled

            # Format for fold
            res = res1 + np.roll(res1,128)
            folded = fold.quickSum(res,256)

            for pos in range(128):
                row = pos
                col = ((rots + pos) % 64)

                
                # 1. Compute which chunk of size 768 to insert
                chunk_pos = row
                # 2. Compute where in chunk to place element
                chunk_offset = (i * 64) + col
                #print(i,rots,pos,chunk_pos)
                # 3. Compute which ciphertext the element belongs in
                cipher_idx = chunk_pos // 16
                # 4. Compute desired_location within cipher
                desired_location = (chunk_pos % 16)*2048 + chunk_offset
                

                #print(i,j,rots,pos,head_row,head_col, abs_pos,desired_location)
                masked_out = mask_out(folded,(pos*256,1))
                
                
                shift_amt = desired_location - pos*256
                rolled = np.roll(masked_out, shift_amt)

                output[cipher_idx] += rolled
    return output*/

/*
def qk_matmul(Q,K):
    #assert Q.shape == (12,32768)
    #assert K.shape == (12,32768)
    
    # Output is (12,128,128) matrices that result from Q^KT
    output = np.zeros((12,32768))

    for i in range(Q.shape[0]):
        # Need to create duplicate format for matmul to work
        # |A|B|C|...|A|B|C|
        dup = K[i] + np.roll(K[i],16384)
        assert np.equal(dup[:16384],dup[16384:]).all()

        for rots in range(128):
            rolled = np.roll(dup,-(rots*128))
            res1 = Q[i] * rolled
            
            # Format for fold
            res = res1 + np.roll(res1,64)
            folded = fold.quickSum(res,128)
            for pos in range(128):
                row = i * 128 + pos
                col = i * 128 + ((rots + pos) % 128)

                abs_pos = row*128 + col
                head_col = abs_pos % 128

                masked_out = mask_out(folded,(pos*128,1))
                
                desired_location = row*256 + head_col
                
                shift_amt = desired_location - pos*128
                rolled = np.roll(masked_out, shift_amt)
                
                #print(i,j,rots,pos,head_row,head_col, abs_pos,desired_location)

                output[i] += rolled
    return output
*/
/*def sv_matmul(S,V):
    print(S.shape)
    assert S.shape == (12,32768) 
    assert V.shape == (12,32768)

    # Output is (8,32768) matrix
    output = np.zeros((8,32768))
    for i in range(12):
        for rots in range(64):

            # Need to create duplicate format for rotates to work
            # |A|B|C|...|A|B|C|
            dup = V[i] + np.roll(V[i],16384)

            rolled = np.roll(dup,-(rots*256))
            res1 = S[i] * rolled

            # Format for fold
            res = res1 + np.roll(res1,128)
            folded = fold.quickSum(res,256)

            for pos in range(128):
                row = pos
                col = ((rots + pos) % 64)

                
                # 1. Compute which chunk of size 768 to insert
                chunk_pos = row
                # 2. Compute where in chunk to place element
                chunk_offset = (i * 64) + col
                #print(i,rots,pos,chunk_pos)
                # 3. Compute which ciphertext the element belongs in
                cipher_idx = chunk_pos // 16
                # 4. Compute desired_location within cipher
                desired_location = (chunk_pos % 16)*2048 + chunk_offset
                

                #print(i,j,rots,pos,head_row,head_col, abs_pos,desired_location)
                masked_out = mask_out(folded,(pos*256,1))
                
                
                shift_amt = desired_location - pos*256
                rolled = np.roll(masked_out, shift_amt)

                output[cipher_idx] += rolled
    return output*/