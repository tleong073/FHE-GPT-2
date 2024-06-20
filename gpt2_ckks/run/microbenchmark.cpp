#include "approx.h"
#include "test_util.h"

#include <chrono>

using namespace std;
using namespace NTL;
using namespace seal;
using namespace chrono;

void benchmark_attn_proj(SEALContext context,KeyGenerator &keygen, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,Evaluator &evaluator,GaloisKeys &gal_keys, RelinKeys &relin_keys) {

    Plaintext plain;
    Ciphertext cipher,zero_cipher;

    // Setup Ingredients
    vvec A1(128,vector<double>(768,1.0));
    vvec A2(768,vector<double>(768,1.0));
    vvec A_t(768,vector<double>(768,0.0));

    vvec A_exp(128,vector<double>(768,0.0));
    vec zero_bias(32768,0.0);
    vvec A2_out(1,vec(32768,0.0));

    generate_random(A1);
    generate_random(A2);

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

    auto start = system_clock::now();
    decrypt_and_print_and_max_round(A1_cipher[0],decryptor,encoder,1.0,0.0);
    attn_proj_row_seal(A1_cipher,A2_cipher,bias,output,128,768,768,128,
                        keygen,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    duration<double> sec = system_clock::now() - start;
    cout << "attn_proj time : " << sec.count() << "s" << endl;

    vec res;
    res.reserve(32768);


    // Correctness check
    int i,j,pos;
    vvec decrypted;
    for(i=0;i<output.size();i++){
        decryptor.decrypt(output[i],plain);
        encoder.decode(plain,res);
        decrypted.push_back(res);
    }
    
    for(i =0; i<128;i++){
        for(j =0;j<768;j++) {
            pos = i*768+j;
            if(A_exp[i][j] != decrypted[pos / 32768][pos % 32768]){
                throw std::system_error();
            }
        }
    }

}

void benchmark_qk_matmul(SEALContext context,KeyGenerator &keygen, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,Evaluator &evaluator,GaloisKeys &gal_keys, RelinKeys &relin_keys) {
    
    Plaintext plain;
    Ciphertext cipher,zero_cipher;

    // Setup Ingredients
    vvec A1(128,vector<double>(768,1.0));
    vvec A2(768,vector<double>(768,1.0));
    vvec A_t(768,vector<double>(768,0.0));

    vvec A_exp(128,vector<double>(768,0.0));
    vec zero_bias(32768,0.0);
    vvec A2_out(1,vec(32768,0.0));

    generate_random(A1);
    generate_random(A2);

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

    auto start = system_clock::now();
    decrypt_and_print_and_max_round(A1_cipher[0],decryptor,encoder,1.0,0.0);
    attn_proj_row_seal(A1_cipher,A2_cipher,bias,output,128,768,768,128,
                        keygen,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    duration<double> sec = system_clock::now() - start;
    cout << "attn_proj time : " << sec.count() << "s" << endl;

    vec res;
    res.reserve(32768);


    // Correctness check
    int i,j,pos;
    vvec decrypted;
    for(i=0;i<output.size();i++){
        decryptor.decrypt(output[i],plain);
        encoder.decode(plain,res);
        decrypted.push_back(res);
    }
    
    for(i =0; i<128;i++){
        for(j =0;j<768;j++) {
            pos = i*768+j;
            if(A_exp[i][j] != decrypted[pos / 32768][pos % 32768]){
                throw std::system_error();
            }
        }
    }

    return;
}

void benchmark_softmax(SEALContext context,KeyGenerator &keygen,Bootstrapper &bootstrapper, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,Evaluator &evaluator,GaloisKeys &gal_keys, RelinKeys &relin_keys,bool optimized) {

    Plaintext plain;
    Ciphertext cipher,zero_cipher,out_cipher;

    vvec v(1,vec(32768,0.0));

    generate_random(v);

    vec out(32768,0.0);

    encoder.encode(v[0], ENCODE_SCALE, plain);
    encryptor.encrypt(plain,cipher);

    while(cipher.coeff_modulus_size() > (TOTAL_LEVEL-BOOT_LEVEL-1)){
        evaluator.mod_switch_to_next_inplace(cipher);
    }
        

    compute_softmax_plain(v[0],out);

    string desc;

    auto start = system_clock::now();

    if(!optimized){
        desc= "Softmax time: ";
        compute_softmax(cipher,6,bootstrapper,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    } else {
        desc= "Smax time: ";
        compute_smax(cipher,6,0.1,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    }

    duration<double> sec = system_clock::now() - start;
    cout << desc << sec.count() << "s" << endl;
    
    decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0);
}

void benchmark_layernorm(SEALContext context,KeyGenerator &keygen,Bootstrapper &bootstrapper, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,Evaluator &evaluator,GaloisKeys &gal_keys, RelinKeys &relin_keys,bool optimized) {

    Plaintext plain;
    Ciphertext cipher,zero_cipher,out_cipher;

    vvec v(1,vec(32768,0.0));

    generate_random(v);

    vec out(32768,0.0);

    encoder.encode(v[0], ENCODE_SCALE, plain);
    encryptor.encrypt(plain,cipher);

    while(cipher.coeff_modulus_size() > (TOTAL_LEVEL-BOOT_LEVEL)){
        evaluator.mod_switch_to_next_inplace(cipher);
    }
        

    compute_softmax_plain(v[0],out);

    string desc;

    vec gamma(768,0.8);
    vec beta(768,0.9);
    auto start = system_clock::now();
    //compute_softmax(cipher,6,bootstrapper,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    compute_layernorm(cipher,out_cipher,gamma,beta,768,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    duration<double> sec = system_clock::now() - start;
    cout << "lnorm_time: " << sec.count() << "s" << endl;
    
    decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0);
}


int main(int argc, char* argv[]) {

    map<int,test_entry_t> tests;

    populate_tests(tests);
    if(argc < 2){
        cout << "Please specify which microbenchmark you want to run. " << endl;
        print_tests(tests);
        return 0;
    }

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

	int logp = LOGP;
	int logq = LOGQ;
	int log_special_prime = 60;

    int log_integer_part = logq - logp - loge + 5;

	// int remaining_level = 14; // Calculation required
	int remaining_level = 21; // Calculation required
	int boot_level = 14; // greater than: subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2 
	int total_level = remaining_level + boot_level;

    INIT();

    Bootstrapper bootstrapper(loge, logn, logN - 1, total_level, scale, boundary_K, deg, scale_factor, inverse_deg, context, keygen, encoder, encryptor, decryptor, evaluator, relin_keys, gal_keys);

    cout << "Generating Optimal Minimax Polynomials..." << endl;
    bootstrapper.prepare_mod_polynomial();
    cout << "Adding Bootstrapping Keys..." << endl;
    //bootstrapper.addBootKeys_3_other_slots(gal_keys, slot_vec);
	bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

    bootstrapper.slot_vec.push_back(logn);

	cout << "Generating Linear Transformation Coefficients..." << endl;
	bootstrapper.generate_LT_coefficient_3();

    

    int entry = atoi(argv[1]);

    /*
    enum TestType {
    ATTN_PROJ_ROW,
    QK_MATMUL,
    SV_MATMUL,
    SOFTMAX,
    SMAX,
    GELU,
    LAYERNORM,
    BOOTSTRAP
    };
    */
    switch(entry) {
        case ATTN_PROJ_ROW:
            cout <<"Executing: " << tests[entry].name <<endl;
            benchmark_attn_proj(context,keygen,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
            cout << "Done Executing " << tests[entry].name << endl;
            break;
        case QK_MATMUL:
            cout <<"Executing: " << tests[entry].name <<endl;
            benchmark_qk_matmul(context,keygen,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
            cout << "Done Executing " << tests[entry].name << endl;
            break;
        case SOFTMAX:
            cout <<"Executing: " << tests[entry].name <<endl;
            benchmark_softmax(context,keygen,bootstrapper,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys,false);
            cout << "Done Executing " << tests[entry].name << endl;
            break;
        case SMAX:
            cout <<"Executing: " << tests[entry].name <<endl;
            benchmark_softmax(context,keygen,bootstrapper,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys,true);
            cout << "Done Executing " << tests[entry].name << endl;
            break;
        case LAYERNORM:
            cout <<"Executing: " << tests[entry].name <<endl;
            benchmark_layernorm(context,keygen,bootstrapper,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys,true);
            cout << "Done Executing " << tests[entry].name << endl;
            break;
        case BOOTSTRAP:
            cout <<"Executing: " << tests[entry].name <<endl;
            benchmark_softmax(context,keygen,bootstrapper,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys,true);
            cout << "Done Executing " << tests[entry].name << endl;
            break;
        default:
            cout << "Invalid entry. Please input a valid test type: " << endl;
            print_tests(tests);
    }


    return 0;
}