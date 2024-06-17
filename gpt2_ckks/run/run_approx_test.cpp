#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <iostream>
#include <algorithm>
#include "approx.h"
#include "pack.h"
#include "test_util.h"


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
int logp = LOGP;
int logq = 51;
int log_special_prime = 60;
int log_integer_part = logq - logp - loge + 5;
int remaining_level = 10; // Calculation required
int boot_level = 14; // 
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

TEST_SUITE("Pack") {

    TEST_CASE("MaskOut") {
        INIT();

        Plaintext plain;
        Ciphertext in;

        vec v= {1.0,2.0,3.0,4.0};
        vector<double> res;
        res.resize(32768,1.0);
        vector<double> ones = {1.0,1.0,1.0,1.0};
        ones.resize(32768,1.0);
        encoder.encode(v,scale,plain);
        encryptor.encrypt(plain,in);

        encoder.encode(ones,scale,plain);
        
        int amt = 512;
        Ciphertext out;
        evaluator.rotate_vector(in,amt,gal_keys,out);
        decrypt_and_print_and_max_round(out,decryptor,encoder,1.0,0);

        decryptor.decrypt(out,plain);
        encoder.decode(plain,res);

        for(int i = 32768-amt; i<32768-amt+4;i++)
            CHECK(doctest::Approx(res[i]) == v[i-(32768-amt)]);
    }

    TEST_CASE("PackFromRow") {

        INIT();
        printf("TESTING PACK FROM ROW\n");

        Plaintext plain;

        vector<Ciphertext> out;
        vector<Ciphertext> out2;

        vector<Plaintext> out_plain;
        vector<double> tmp_vector;
        vector<vector<double>> res;

        // Init output vectors
        init_output(8,out,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        init_output(3,out2,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

        // Init test matrix
        vector<vector<double>> A(128,vector<double>(768,0.0));
        for(int i = 0;i<128;i++) {
            for(int j = 0; j<768;j++) {
                A[i][j] = i*768+j;
            }
        }

        // Init exp matrix
        vector<vector<double>> A_exp(3,vector<double>(32768,0.0));
        for(int i = 0;i<3;i++) {
            for(int j = 0; j<32768;j++) {
                A_exp[i][j] = i*32768+j;
            }
        }

        printf("Done with init\n");
        decrypt_and_print_and_max_round(out[0],decryptor,encoder,1.0,0);
        // Pack from row
        pack_from_row(A,out,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        
        printf("Done packing from row\n");
        decrypt_and_print_and_max_round(out[0],decryptor,encoder,1.0,0);

        // Pack tight
        pack_tight(out,out2,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

        printf("Done packing tight\n");

        for(int i = 0; i<3;i++) {
            decryptor.decrypt(out2[i],plain);
            encoder.decode(plain,tmp_vector);
            res.push_back(tmp_vector);
        }

        // Check correctness
        for(int i = 0; i<3;i++) {
            for(int j = 0; j<32768;j++) {
                CHECK(doctest::Approx(res[i][j]) == A_exp[i][j]);
            }
        }
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
        Ciphertext cipher,zero_cipher;

        // Setup Ingredients
        vvec A1(8,vector<double>(2048,1.0));
        vvec A2(8,vector<double>(2048,1.0));
        vvec A_t(2048,vector<double>(8,1.0));


        vvec A_exp(8,vector<double>(8,0.0));
        vec zero_bias(32768,0.0000001);

        encoder.encode(zero_bias,ENCODE_SCALE,plain);
        encryptor.encrypt(plain,zero_cipher);

        printf("ABOUT TO PLAIN MATMUL\n");
        transpose(A2,A_t);
        matrix_mul(A1,A_t,A_exp);

        printf("ABOUT TO INIT OUTPUT\n");
        vector<Ciphertext>A1_cipher,A2_cipher,output,pack_out;
        //init_output(1,A1_cipher,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        //init_output(1,A2_cipher,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        init_output(1,output,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

        //pack_from_row(A1,A1_cipher,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        //pack_from_row(A2,A2_cipher,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        vvec A1_pre(1,vec(32768,0.0));
        vvec A2_pre(1,vec(32768,0.0));

        pack_plain_row(A1,8,2048,A1_pre);
        pack_plain_row(A2,8,2048,A2_pre);

        encoder.encode(A1_pre[0],ENCODE_SCALE,plain);
        encryptor.encrypt(plain,cipher);
        A1_cipher.push_back(cipher);

        decrypt_and_print_and_max_round(A1_cipher[0],decryptor,encoder,1.0,0);

        encoder.encode(A2_pre[0],ENCODE_SCALE,plain);
        encryptor.encrypt(plain,cipher);
        A2_cipher.push_back(cipher);


        printf("Done packing into ciphertexts\n");
       

        Config conf;
        row_matrix_multiplication_seal(A1_cipher,A2_cipher,zero_cipher,output,8,2048,2048,8,
                        encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        printf("Done with matmul\n");

        int global_idx=0;
        vec res;
        decryptor.decrypt(output[0],plain);
        encoder.decode(plain,res);
        for(int i =0; i<8;i++){
            for(int j =0;j<8;j++) {
                CHECK(doctest::Approx(res[i*16+j]) == A_exp[i][j]);
            }
        }
        
        
        
    }
    TEST_CASE("AttnProjRow") {
        // Test 

        INIT();
        Plaintext plain;
        Ciphertext cipher,zero_cipher;

        // Setup Ingredients
        vvec A1(16,vector<double>(1024,1.0));
        vvec A2(1024,vector<double>(16,1.0));
        vvec A_t(16,vector<double>(1024,0.0));

        vvec A_exp(16,vector<double>(16,0.0));
        vec zero_bias(32768,0.0000001);
        vvec A2_out(1,vec(32768,0.0));

        for(int i=0; i<1024;i++){
            for(int j=0;j<16;j++) {
                //printf("%d %f %f ",j,(double)(i*768+j),A2[i][j]);
                A2[i][j] = (double)(16*i+j);
                printf("%f\n",i,A2[i][j]);
            }
        }

        printf("ABOUT TO PLAIN MATMUL\n");
        transpose(A2,A_t);
        matrix_mul(A1,A2,A_exp);
        
        printf("ABOUT TO INIT OUTPUT\n");
        vector<Ciphertext>A1_cipher,A2_cipher,output,pack_out;
        //init_output(1,A1_cipher,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        //init_output(48,A2_cipher,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        init_output(12,output,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

        printf("Packing from Row\n");
        pack_from_row(A1,A1_cipher,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        pack_from_row(A_t,A2_cipher,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        
        printf("Done packing into ciphertexts: A1: %zu  A2: %zu\n",A1_cipher.size(),A2_cipher.size());

        Config conf;
        vec zeros(32768,0.0);
        Ciphertext bias;
        encoder.encode(zeros,ENCODE_SCALE,plain);
        encryptor.encrypt(plain,bias);
        attn_proj_row_seal(A1_cipher,A2_cipher,bias,output,16,1024,1024,16,
                                encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        printf("Done with matmul\n");

        int global_idx=0;
        vec res;
        decryptor.decrypt(output[0],plain);
        encoder.decode(plain,res);
        for(int i =0; i<1;i++){
            for(int j =0;j<16;j++) {
                CHECK(doctest::Approx(res[j*16+(j*17)]) == A_exp[i][j]);
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

        vector<double> v = {0.0035,0.4,0.67,2.23284};
        v.resize(32768,2.0);
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

        vector<double> v = {20,100,0.05,25};
        encoder.encode(v, scale, plain);
        encryptor.encrypt(plain,cipher_in);

        compute_inv_sqrt(cipher_in,cipher_out,3,0.1, encoder, encryptor, decryptor, evaluator, gal_keys, relin_keys);
        
        vector<double> v_expect = {0.21299612278784003,0.09999999999999999,0.337032947479256,0.2};
        vector<double> v_out;
        decryptor.decrypt(cipher_out,plain);
        encoder.decode(plain, v_out);
        
        for(int i = 0; i<v.size();i++) {
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
        Ciphertext cipher1,cipher2;

        TensorCipher tensor1;
        TensorCipher tensor2;

        vector<double> v = {1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8};
        encoder.encode(v, scale, plain);

        encryptor.encrypt(plain,cipher1);

        quickSum(cipher1,cipher2,8,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

        vector<double> v_expect  = vector<double>(8,36.0),v_res;
        decryptor.decrypt(cipher2,plain);
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
        Ciphertext cipher,out_cipher;

        TensorCipher tensor1;
        TensorCipher tensor2;

        vector<double> v = {.1,.2,.3,.4,.5,.6,.7,.8,.1,.2,.3,.4,.5,.6,.7,.8};
        encoder.encode(v, scale, plain);

        encryptor.encrypt(plain,cipher);


        quickMax(cipher,out_cipher,8,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

        vector<double> v_expect  = vector<double>(8,0.8),v_res;
        decryptor.decrypt(out_cipher,plain);
        encoder.decode(plain, v_res);
        
        for(int i = 0; i<v_expect.size();i++) {
            CHECK(doctest::Approx(v_res[i]) == v_expect[i]);
        }
    }

    TEST_CASE("Softmax") {

        INIT();

        Plaintext plain;
        Ciphertext cipher,out_cipher;

        vector<double> v = {1,2,3,4,5};
        vec out(5,0.0);
        encoder.encode(v, scale, plain);
        encryptor.encrypt(plain,cipher);

        compute_softmax_plain(v,out);

        compute_softmax(cipher,6,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

        decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0);

    }

    TEST_CASE("LayerNorm") {

        INIT();

        Plaintext plain;
        Ciphertext cipher,out_cipher;

        vector<double> v = {.1,.2,.3,.4,.5};
        vec out(5,0.0);
        encoder.encode(v, scale, plain);
        encryptor.encrypt(plain,cipher);

        compute_softmax_plain(v,out);
        vec gamma(768,1.0);
        vec beta(768,1.0);

        compute_layernorm(cipher,out_cipher,gamma,beta,768,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

        decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0);

    }

}

TEST_SUITE("Bootstrap") {

    TEST_CASE("SingleBootstrap") {

        INIT();

        Plaintext plain;
        Ciphertext cipher,rtn;

        vector<double> v = {0.4,0.5};
        encoder.encode(v, scale, plain);

        encryptor.encrypt(plain,cipher);
        int i;
        for(i=0;i<total_level;i++){
            evaluator.square_inplace(cipher);
            evaluator.relinearize_inplace(cipher,relin_keys);
            evaluator.rescale_to_next_inplace(cipher);
            printf("Hello: %zu\n",cipher.coeff_modulus_size());
        }
        

        Bootstrapper bootstrapper_1(loge, logn_1, logN - 1, total_level, scale, boundary_K, boot_deg, scale_factor, inverse_deg, context, keygen, encoder, encryptor, decryptor, evaluator, relin_keys, gal_keys);
        bootstrapper_1.prepare_mod_polynomial();
        bootstrapper_1.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);
        bootstrapper_1.generate_LT_coefficient_3();
        bootstrapper_1.bootstrap_real_3(rtn,cipher);

        vector<double> v_mod;
        decryptor.decrypt(rtn,plain);
        encoder.decode(plain, v_mod);
        
        CHECK(v[0]*v[0] == doctest::Approx(v_mod[0]));
        CHECK(v[1]*v[1] == doctest::Approx(v_mod[1]));
    }

}