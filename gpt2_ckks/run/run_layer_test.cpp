#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <iostream>
#include <algorithm>
#include "gpt2_seal.h"

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

TEST_SUITE("Attention") {

    TEST_CASE("AttentionLayer") {

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