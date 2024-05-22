#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <iostream>
#include <algorithm>
#include "approx.h"


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
int logp = 51;
int logq = 51;
int log_special_prime = 51;
int log_integer_part = logq - logp - loge + 5;
int remaining_level = 9; // Calculation required
int boot_level = 1; // 
int total_level = remaining_level + boot_level;



TEST_SUITE("Init") {

    TEST_CASE("InitNoCrash") {

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

        vector<double> v2 = {4.0,5.0,6.0,4.0,5.0,6.0};
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
        vector<vector<double>> expected;
        expected.push_back({4.0,10.0,18.0});
        expected.push_back({5.0,12.0,12.0});
        expected.push_back({6.0,8.0,15.0});


        col_matrix_multiplication_seal(left_inputs, right_inputs, outputs, bias, rows, cols, config, 
            encoder, encryptor, decryptor, evaluator, gal_keys, relin_keys);
        
        for(int i =0; i <cols; i++) {
            decryptor.decrypt(outputs[i].cipher(),plain);
            encoder.decode(plain, output_unencoded);
            for(int j = 0;j<cols;j++) {
                //printf("%d ",output_unen);
                CHECK(doctest::Approx(output_unencoded[j]) == expected[i][j]);
            }
            //printf("\n");
        }
        
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

    TEST_CASE("DiagRowMat") {

        INIT();
        Plaintext plain;
        Ciphertext cipher;

        vector<vector<double>> inputs;
        vector<TensorCipher> tmp_tcipher;
        vector<vector<TensorCipher>> left_inputs;
        vector<TensorCipher> inputs_encrypted;
        TensorCipher tensor;
        Config config;
        
        size_t slot_count = encoder.slot_count();
        cout << "Number of slots: " << slot_count << endl;


        vector<double> w1 = {1.0,2.0,3.0};
        vector<double> w2 = {4.0,5.0,6.0};
        vector<double> w3 = {7.0,8.0,9.0};
        inputs.push_back(w1);
        inputs.push_back(w2);
        inputs.push_back(w3);

        int rows = 3,cols = 3;
        for(int i = 0; i < rows; i++){
            encoder.encode(inputs[i], scale, plain);
            encryptor.encrypt(plain, cipher);
            tensor = TensorCipher(cipher);
            inputs_encrypted.push_back(tensor);
        }

        printf("Setup Complete: \n");
        vector<TensorCipher> outputs;
        vector<double> output_unencoded;

        diagonal_to_row_matrix_seal( inputs_encrypted,outputs, rows,  cols,config,
	         encoder, encryptor, decryptor, evaluator, gal_keys, relin_keys);
        
        vector<vector<double>> expect;
        expect.push_back({1.0,4.0,7.0});
        expect.push_back({2.0,5.0,8.0});
        expect.push_back({3.0,6.0,9.0});

        int idx;
        printf("Decrypting\n");
        for(int i = 0; i<rows;i++) {
            decryptor.decrypt(outputs[i].cipher(),plain);
            encoder.decode(plain, output_unencoded);
            for(int j =0 ; j< cols;j++) {
                CHECK(doctest::Approx(output_unencoded[j]) == expect[i][j]);
            }
        }
    }
}

TEST_SUITE("PolyApprox") {

    TEST_CASE("SignFunctionF") {

        INIT();

        Plaintext plain;
        Ciphertext cipher_in,cipher_out;

        vector<double> v = {-0.4000000,0.5000000000,-1,1};
        encoder.encode(v, scale, plain);
        encryptor.encrypt(plain,cipher_in);

        vector<Ciphertext> cheby_basis;

        compute_sign_f(cipher_in,cipher_out, encoder, encryptor, decryptor, evaluator, gal_keys, relin_keys);
        
        vector<double> v_expect = {-0.80238268,0.9021453857,-1.0,1.0};
        vector<double> v_out;
        decryptor.decrypt(cipher_out,plain);
        encoder.decode(plain, v_out);
        
        for(int i = 0; i<4;i++) {
             CHECK(doctest::Approx(v_out[i]) == v_expect[i]);
        }
    }

    TEST_CASE("SignFunctionG") {

        INIT();

        Plaintext plain;
        Ciphertext cipher_in,cipher_out;

        vector<double> v = {-0.4000000,0.5000000000,-1,1};
        encoder.encode(v, scale, plain);
        encryptor.encrypt(plain,cipher_in);

        vector<Ciphertext> cheby_basis;

        compute_sign_g(cipher_in,cipher_out, encoder, encryptor, decryptor, evaluator, gal_keys, relin_keys);
        
        vector<double> v_expect = {-0.899779538,0.7708721161,-0.998046875,0.998046875};
        vector<double> v_out;
        decryptor.decrypt(cipher_out,plain);
        encoder.decode(plain, v_out);
        
        for(int i = 0; i<4;i++) {
             CHECK(doctest::Approx(v_out[i]) == v_expect[i]);
        }
    }

    TEST_CASE("SignFunction") {

        INIT();

        Plaintext plain;
        Ciphertext cipher_in,cipher_out;
        TensorCipher tensor;

        vector<double> v = {-0.4000000,0.5000000000,0.01,-0.02};
        encoder.encode(v, scale, plain);
        encryptor.encrypt(plain,cipher_in);

        TensorCipher t = TensorCipher(cipher_in);

        sign_function(t,t,2,2, encoder, encryptor, decryptor, evaluator, gal_keys, relin_keys);
        
        vector<double> v_expect = {-1,1,0.98683881,-0.9999994};
        vector<double> v_out;
        decryptor.decrypt(t.cipher(),plain);
        encoder.decode(plain, v_out);
        
        for(int i = 0; i<4;i++) {
             CHECK(doctest::Approx(v_out[i]) == v_expect[i]);
        }
    }

    TEST_CASE("GeluP") {

        INIT();

        Plaintext plain;
        Ciphertext cipher_in,cipher_out;

        vector<double> v = {-0.4,0.5,1,-1};
        encoder.encode(v, scale, plain);
        encryptor.encrypt(plain,cipher_in);

        compute_gelu_p(cipher_in,cipher_out, encoder, encryptor, decryptor, evaluator, gal_keys, relin_keys);
        
        vector<double> v_expect = {-0.3501723443,-0.7345966621,-1.036827125,-0.188669242};
        vector<double> v_out;
        decryptor.decrypt(cipher_out,plain);
        encoder.decode(plain, v_out);
        
        for(int i = 0; i<4;i++) {
            CHECK(doctest::Approx(v_out[i]) == v_expect[i]);
        }
    }

    TEST_CASE("GeluQ") {

        INIT();

        Plaintext plain;
        Ciphertext cipher_in,cipher_out;

        vector<double> v = {-3,5,1,-1};
        encoder.encode(v, scale, plain);
        encryptor.encrypt(plain,cipher_in);

        compute_gelu_q(cipher_in,cipher_out, encoder, encryptor, decryptor, evaluator, gal_keys, relin_keys);
        
        vector<double> v_expect = {-0.5845409261,13.43445935,0.8339413477,-0.1655280783};
        vector<double> v_out;
        decryptor.decrypt(cipher_out,plain);
        encoder.decode(plain, v_out);
        
        for(int i = 0; i<4;i++) {
            CHECK(doctest::Approx(v_out[i]) == v_expect[i]);
        }
    }
}

TEST_SUITE("IterApprox") {

    TEST_CASE("Goldschmidt") {

        INIT();

        Plaintext plain;
        Ciphertext cipher_in,cipher_out;

        vector<double> v = {0.0035,0.4,0.67};
        encoder.encode(v, scale, plain);
        encryptor.encrypt(plain,cipher_in);

        compute_inverse(cipher_in,cipher_out,8, encoder, encryptor, decryptor, evaluator, gal_keys, relin_keys);
        
        vector<double> v_expect = {169.26910130445748,2.499999999999999,1.4925373134328357};
        vector<double> v_out;
        decryptor.decrypt(cipher_out,plain);
        encoder.decode(plain, v_out);
        
        for(int i = 0; i<3;i++) {
             CHECK(doctest::Approx(v_out[i]) == v_expect[i]);
        }
    }

    TEST_CASE("Newton") {

        INIT();

        Plaintext plain;
        Ciphertext cipher_in,cipher_out;

        vector<double> v = {20,100,0.05};
        encoder.encode(v, scale, plain);
        encryptor.encrypt(plain,cipher_in);

        compute_inv_sqrt(cipher_in,cipher_out,3,0.1, encoder, encryptor, decryptor, evaluator, gal_keys, relin_keys);
        
        vector<double> v_expect = {0.21299612278784003,0.09999999999999999,0.337032947479256};
        vector<double> v_out;
        decryptor.decrypt(cipher_out,plain);
        encoder.decode(plain, v_out);
        
        for(int i = 0; i<3;i++) {
             CHECK(doctest::Approx(v_out[i]) == v_expect[i]);
        }
    }

    TEST_CASE("Exp") {

        INIT();

        Plaintext plain;
        Ciphertext cipher_in,cipher_out;

        vector<double> v = {2,-0.05,10};
        encoder.encode(v, scale, plain);
        encryptor.encrypt(plain,cipher_in);

        compute_exp(cipher_in,cipher_out,6, encoder, encryptor, decryptor, evaluator, gal_keys, relin_keys);
        
        vector<double> v_expect = {7.166276152788219,0.9512108363005606,10847.05214173728};
        vector<double> v_out;
        decryptor.decrypt(cipher_out,plain);
        encoder.decode(plain, v_out);
        
        for(int i = 0; i<3;i++) {
             CHECK(doctest::Approx(v_out[i]) == v_expect[i]);
        }
    }
}

TEST_SUITE("Fold") {

    TEST_CASE("QuickSum") {

        INIT();

        Plaintext plain;
        Ciphertext cipher;

        TensorCipher tensor1;
        TensorCipher tensor2;

        vector<double> v = {1,2,3,4,5,6,7,8};
        encoder.encode(v, scale, plain);

        encryptor.encrypt(plain,cipher);

        tensor1 = TensorCipher(cipher);

        quickSum(tensor1,tensor2,8,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

        vector<double> v_expect  = vector<double>(8,36.0),v_res;
        decryptor.decrypt(tensor2.cipher(),plain);
        encoder.decode(plain, v_res);
        
        for(int i = 0; i<v_expect.size();i++) {
            CHECK(doctest::Approx(v_res[i]) == v_expect[i]);
        }
    }

    TEST_CASE("ComputeMax") {

        INIT();

        Plaintext plain;
        Ciphertext cipher1,cipher2,out_cipher;

        TensorCipher tensor1;
        TensorCipher tensor2;

        vector<double> v1 = {0.1,0.5,0.003,0.4,-0.2};
        vector<double> v2 = {0.3,0.1,0.1,-0.6,0.0001};
        encoder.encode(v1, scale, plain);
        encryptor.encrypt(plain,cipher1);

        encoder.encode(v2, scale, plain);
        encryptor.encrypt(plain,cipher2);

        computeMax(cipher1,cipher2,out_cipher,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

        vector<double> v_expect  = {0.3,0.5,0.1,0.4,0.0001};
        vector<double> v_res;
        decryptor.decrypt(out_cipher,plain);
        encoder.decode(plain, v_res);
        
        for(int i = 0; i<v_expect.size();i++) {
            CHECK(doctest::Approx(v_res[i]) == v_expect[i]);
        }
    }

    TEST_CASE("QuickMax") {

        INIT();

        Plaintext plain;
        Ciphertext cipher;

        TensorCipher tensor1;
        TensorCipher tensor2;

        vector<double> v = {.1,.2,.3,.4,.5,.6,.7,.8};
        encoder.encode(v, scale, plain);

        encryptor.encrypt(plain,cipher);

        tensor1 = TensorCipher(cipher);

        quickMax(tensor1,tensor2,8,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

        vector<double> v_expect  = vector<double>(8,0.8),v_res;
        decryptor.decrypt(tensor2.cipher(),plain);
        encoder.decode(plain, v_res);
        
        for(int i = 0; i<v_expect.size();i++) {
            CHECK(doctest::Approx(v_res[i]) == v_expect[i]);
        }
    }

}