#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <iostream>
#include <algorithm>
#include "gpt2_seal.h"



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


// SEAL and bootstrapping setting
long boundary_K = 25;
long boot_deg = 59;
long scale_factor = 2;
long inverse_deg = 1; 
long logN = 16;
long loge = 10; 
long logn = 15;		// full slots
long logn_1 = 14;	// sparse slots
long logn_2 = 13;
long logn_3 = 12;
int logp = 46;
int logq = 51;
int log_special_prime = 51;
int log_integer_part = logq - logp - loge + 5;
int remaining_level = 5; // Calculation required
int boot_level = 1; // 
int total_level = remaining_level + boot_level;



TEST_SUITE("Init") {

    TEST_CASE("Setup initializes correctly") {

        INIT();

        Plaintext plain;
        Ciphertext cipher1;
        Ciphertext cipher2;
        Ciphertext cipher3;

        TensorCipher tensor1;
        TensorCipher tensor2;

        vector<double> v = {0.4,0.5};
        encoder.encode(v, scale, plain);

        encryptor.encrypt(plain,cipher1);
        encryptor.encrypt(plain,cipher2);

        tensor1.set_ciphertext(cipher1);
        tensor2.set_ciphertext(cipher2);

        evaluator.multiply(tensor1.cipher(),tensor2.cipher(),cipher3);
        evaluator.relinearize_inplace(cipher3, relin_keys);
        evaluator.rescale_to_next_inplace(cipher3);

        vector<double> v_mod;
        decryptor.decrypt(cipher3,plain);
        encoder.decode(plain, v_mod);
        
        CHECK(v[0]*v[0] == doctest::Approx(v_mod[0]));
        CHECK(v[1]*v[1] == doctest::Approx(v_mod[1]));
    }

}


TEST_SUITE("MatrixMul") {

    TEST_CASE("ColMatMul") {

        INIT();
        Plaintext plain;
        Ciphertext cipher;

        vector<vector<double>> left_input_init, right_input_init;
        vector<TensorCipher> left_inputs, right_inputs;
        TensorCipher tensor;
        Config config;
        
        size_t slot_count = encoder.slot_count();
        cout << "Number of slots: " << slot_count << endl;

        vector<double> v1 = {1.0,2.0,3.0};
        left_input_init.push_back(v1);

        vector<double> v2 = {4.0,5.0,6.0};
        right_input_init.push_back(v2);

        int rows = 1,cols = 3;
        for(int i = 0; i < rows; i++){
            
            encoder.encode(left_input_init[i], scale, plain);
            encryptor.encrypt(plain, cipher);
            
            tensor = TensorCipher();
            tensor.set_ciphertext(cipher);
            left_inputs.push_back(tensor);
            
            encoder.encode(right_input_init[i], scale, plain);
            encryptor.encrypt(plain, cipher);

            tensor = TensorCipher();
            tensor.set_ciphertext(cipher);
            right_inputs.push_back(tensor);
        }

        
        vector<TensorCipher> outputs;
        vector<double> output_unencoded;
        vector<double> bias;

        col_matrix_multiplication_seal(left_inputs, right_inputs, outputs, bias, rows, cols, config, 
            encoder, encryptor, decryptor, evaluator, gal_keys, relin_keys);
        
        decryptor.decrypt(outputs[0].cipher(),plain);
        encoder.decode(plain, output_unencoded);
        CHECK(doctest::Approx(output_unencoded[0]) == 4.0);
        CHECK(doctest::Approx(output_unencoded[1]) == 10.0);
        CHECK(doctest::Approx(output_unencoded[2]) == 18.0);
    }

    TEST_CASE("RowMatMul") {

        INIT();
        Plaintext plain;
        Ciphertext cipher;

        vector<vector<double>> left_input_init, weights;
        vector<TensorCipher> tmp_tcipher;
        vector<vector<TensorCipher>> left_inputs;
        TensorCipher tensor;
        Config config;
        
        size_t slot_count = encoder.slot_count();
        cout << "Number of slots: " << slot_count << endl;

        vector<double> v1 = {1.0,2.0,3.0};
        vector<double> v2 = {4.0,5.0,6.0};
        left_input_init.push_back(v1);
        left_input_init.push_back(v2);

        vector<double> w1 = {1.0,2.0,3.0};
        vector<double> w2 = {4.0,5.0,6.0};
        vector<double> w3 = {7.0,8.0,9.0};
        weights.push_back(w1);
        weights.push_back(w2);
        weights.push_back(w3);

        int rows = 2,cols = 3;
        for(int i = 0; i < rows; i++){
            tmp_tcipher.clear();
            for(int j = 0; j< cols;j++) {
                // Each element gets expanded.
                vector<double> v(cols,left_input_init[i][j]);
                encoder.encode(v, scale, plain);
                encryptor.encrypt(plain, cipher);

                tensor = TensorCipher(cipher);
                tmp_tcipher.push_back(tensor);
            }
            left_inputs.push_back(tmp_tcipher);
        }

        printf("Setup Complete: \n");
        vector<TensorCipher> outputs;
        vector<double> output_unencoded;
        vector<double> bias;

        row_matrix_multiplication_seal(left_inputs, weights, outputs, bias, rows, cols, config, 
            encoder, encryptor, decryptor, evaluator, gal_keys, relin_keys);
        
        vector<vector<double>> expect;
        expect.push_back({30.0,36.0,42.0});
        expect.push_back({66.0,81.00,96.00});

        for(int i = 0; i<rows;i++) {
            decryptor.decrypt(outputs[i].cipher(),plain);
            encoder.decode(plain, output_unencoded);
            for(int j =0 ; j< cols;j++) {
                printf("%f ",output_unencoded[j]);
                CHECK(doctest::Approx(output_unencoded[j]) == expect[i][j]);
            }
            printf("\n");
        }
    }
}