#include "approx.h"
#include <cassert>


/**
 * @brief Builds a log chebyshev basis up to 2N
 *		
 * Uses log(N)
 * 
 * 
 * @param input The ciphertext to build the basis over
 * @param chebyBasis The ciphertexts to store the basis in. Assumed empty.
 */
void build_cheby_basis(Ciphertext &input, vector<Ciphertext> &chebyBasis, int n, CKKSEncoder &encoder,
                        Encryptor &encryptor, Decryptor &decryptor, Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys)
{

    assert(n > 0);
	size_t slots = encoder.slot_count();

    //printf("ChebyBasis start\n");
	Evaluator *eval;

	Ciphertext cipher,tmp_cipher;
	Plaintext plain;
	TensorCipher tensor;
	double init_scale = input.scale();

	// Encode 1's for first polynomial T0
	vector<double> ones(slots,1.0);
	encoder.encode(ones,init_scale,plain);
	encryptor.encrypt(plain,cipher);
	
	chebyBasis.push_back(cipher);
    chebyBasis.push_back(input);
    cipher = input;
    printf("input scale+level: %f %zu\n",input.scale(),input.coeff_modulus_size());

    // 1 less depth used for t2 calculation
    evaluator.square_inplace(cipher);
    evaluator.relinearize_inplace(cipher,relin_keys);
    evaluator.rescale_to_next_inplace(cipher);

    printf("line42 T2 scale+level: %f %zu %f %zu\n",cipher.scale(),cipher.coeff_modulus_size(),input.scale(),input.coeff_modulus_size());
    /*
    evaluator.multiply_const_inplace(cipher,2.0);
    evaluator.rescale_to_next_inplace(cipher);
    */
   evaluator.add_inplace(cipher,cipher);

    evaluator.add_const_inplace(cipher,-1.0);
    chebyBasis.push_back(cipher); // L-2
    printf("ChebyBasis T2 done\n");
    printf("line52 T2 scale+level: %f %zu %f %zu\n",cipher.scale(),cipher.coeff_modulus_size(),input.scale(),input.coeff_modulus_size());

   // printf("line 60 T2 scale+level: %f %zu %f %zu\n",tmp_cipher.scale(),tmp_cipher.coeff_modulus_size(),cipher.scale(),cipher.coeff_modulus_size());
    Ciphertext tmp2;

    // 2x
    evaluator.add(input,input,tmp2);

    // 2x* Tn-1
    evaluator.multiply_reduced_error(tmp2,cipher,relin_keys,tmp_cipher);
    // evaluator.relinearize_inplace(tmp_cipher,relin_keys);
    evaluator.rescale_to_next_inplace(tmp_cipher);
    // printf("line 65 T2 scale+level: %f %zu %f %zu\n",tmp_cipher.scale(),tmp_cipher.coeff_modulus_size(),cipher.scale(),cipher.coeff_modulus_size());

    
    evaluator.multiply_const(input,-1.0,tmp2);
    evaluator.rescale_to_next_inplace(tmp2);

    printf("T2 scale+level: %f %zu %f %zu\n",tmp_cipher.scale(),tmp_cipher.coeff_modulus_size(),tmp2.scale(),tmp2.coeff_modulus_size());
    //decrypt_and_print_and_max_round(tmp2,decryptor,encoder,1.0,0,5,5);
    evaluator.add_inplace_reduced_error(tmp_cipher,tmp2);
    chebyBasis.push_back(tmp_cipher); // L-4
    
    //printf("ChebyBasis T3 done\n");

	// Push back T1-Tn
	for(int i = 0; i<n-2;i++) {
		// 2T_{n}^2 - 1
        printf("Pre square %f %zu\n",cipher.scale(),cipher.coeff_modulus_size());
		evaluator.square_inplace(cipher);
		evaluator.relinearize_inplace(cipher,relin_keys);
		evaluator.rescale_to_next_inplace(cipher);
        printf("Post square %f %zu\n",cipher.scale(),cipher.coeff_modulus_size());
        //decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0,5,5);
        /*
		evaluator.multiply_const_inplace(cipher,2.0);
		evaluator.rescale_to_next_inplace(cipher);
        */
        evaluator.add_inplace(cipher,cipher);

		evaluator.add_const_inplace(cipher,-1.0);
        //printf("Post Add %f %zu\n",cipher.scale(),cipher.coeff_modulus_size());
        //printf("Post add %zu\n",cipher.coeff_modulus_size());
        //decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0,5,5);
        chebyBasis.push_back(cipher);
	}
    //printf("ChebyBasis complete\n");
}

void compute_sign_f(Ciphertext &input,Ciphertext &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys)
{
    // Initial level: L
    Plaintext plain;
    Ciphertext cipher,tmp_cipher;
    // Compute Cheby basis
    vector<Ciphertext> cheby_basis;
    build_cheby_basis(input,cheby_basis,4,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    // Compute first level.
    double coeff_fq1 = -0.6767578125;
    double coeff_fr1 =  1.563049316;

    //decrypt_and_print_and_max_round(input,decryptor,encoder,1.0,0,5,5);
    output = input;
    //printf("First level complete0: %f %zu %f %zu\n",output.scale(),output.coeff_modulus_size(),cheby_basis[2].scale(),cheby_basis[2].coeff_modulus_size());
    evaluator.multiply_const(input,coeff_fq1,output);
    //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0,5,5);
    evaluator.rescale_to_next_inplace(output); // L+1
    //decrypt_and_print_and_max_round(cheby_basis[2],decryptor,encoder,1.0,0,5,5);
    //printf("First level complete1: %f %zu %f %zu\n",output.scale(),output.coeff_modulus_size(),cheby_basis[2].scale(),cheby_basis[2].coeff_modulus_size());

    //printf("First level complete2: %f %zu %f %zu\n",output.scale(),output.coeff_modulus_size(),cheby_basis[2].scale(),cheby_basis[2].coeff_modulus_size());

    evaluator.multiply_inplace_reduced_error(output,cheby_basis[2],relin_keys);
    evaluator.rescale_to_next_inplace(output); // L+2

    //printf("fq1 done\n");
    //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0,5,5);
    evaluator.multiply_const(input,coeff_fr1,cipher);
    //printf("First level complete1: %f %zu %f %zu\n",output.scale(),output.coeff_modulus_size(),cipher.scale(),cipher.coeff_modulus_size());
    evaluator.rescale_to_next_inplace(cipher);
    

    //printf("First level complete1: %f %zu %f %zu\n",output.scale(),output.coeff_modulus_size(),cipher.scale(),cipher.coeff_modulus_size());
    evaluator.add_inplace_reduced_error(output,cipher); // L+2 
    //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0,5,5);
    //printf("First level complete: %f %zu\n",output.scale(),output.coeff_modulus_size());

    // Compute second level.

    double coeff_frq2_q = -0.02685546875;
    double coeff_frq2_r = 0.1384277344;

    // Compute frq2_q
    //decrypt_and_print_and_max_round(cheby_basis[3],decryptor,encoder,1.0,0,5,5);
    evaluator.multiply_const(cheby_basis[3],coeff_frq2_q,cipher);
    evaluator.rescale_to_next_inplace(cipher); // L+2
    //printf("frq2 done!\n");

    evaluator.multiply_const(input,coeff_frq2_r,tmp_cipher);//L+1
    //printf("fq2_q_0 done! %f %zu %f %zu\n",cipher.scale(),cipher.coeff_modulus_size(),tmp_cipher.scale(),tmp_cipher.coeff_modulus_size());
    evaluator.rescale_to_next_inplace(tmp_cipher);

    //printf("fq2_q_0 done! %f %zu %f %zu\n",cipher.scale(),cipher.coeff_modulus_size(),tmp_cipher.scale(),tmp_cipher.coeff_modulus_size());
    evaluator.add_inplace_reduced_error(cipher,tmp_cipher); // L+2
    //decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0,5,5);
    //printf("f2q done! %f %zu %f %zu\n",cipher.scale(),cipher.coeff_modulus_size(),cheby_basis[3].scale(),cheby_basis[3].coeff_modulus_size());
    //printf("modulus check 3 %f %zu %f %zu\n",cipher.scale(),cipher.coeff_modulus_size(),cheby_basis[4].scale(),cheby_basis[4].coeff_modulus_size());
    //decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0,5,5);

    //printf("modulus check %f %zu %f %zu\n",cipher.scale(),cipher.coeff_modulus_size(),tmp_cipher.scale(),tmp_cipher.coeff_modulus_size());
    evaluator.multiply_inplace_reduced_error(cipher,cheby_basis[4],relin_keys); // L+3
    //evaluator.relinearize_inplace(cipher,relin_keys);
    evaluator.rescale_to_next_inplace(cipher); // L+4
    //printf("fq2_q done! %f %zu\n",cipher.scale(),cipher.coeff_modulus_size());
    //decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0,5,5);
    
    //evaluator.mod_switch_to_inplace(output,cipher.parms_id());
    //printf("fq3_q done! %f %zu %f %zu\n",output.scale(),output.coeff_modulus_size(),cipher.scale(),cipher.coeff_modulus_size());
    // Compute fr3 = frq2*cheby[3] + fr2
    //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0,5,5);
    evaluator.add_inplace_reduced_error(output,cipher); // L+4
    
    //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0,5,5);
   

    // Compute 3rd and final level
    double coeff_fq3 = 0.002136230469;

    evaluator.multiply_const(input,coeff_fq3,cipher);
    evaluator.rescale_to_next_inplace(cipher);
    

   // printf("fq4 done! %f %zu %f %zu\n",cheby_basis[5].scale(),cheby_basis[5].coeff_modulus_size(),cipher.scale(),cipher.coeff_modulus_size());

    evaluator.multiply_inplace_reduced_error(cipher,cheby_basis[5],relin_keys);
    evaluator.rescale_to_next_inplace(cipher); // L+6

    

    //printf("fq5 done! %f %zu %f %zu\n",output.scale(),output.coeff_modulus_size(),cipher.scale(),cipher.coeff_modulus_size());
    
   
    //decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0,5,5);
    //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0,5,5);
    

    evaluator.add_inplace_reduced_error(output,cipher);


    return;
}

void compute_sign_g(Ciphertext &input,Ciphertext &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys)
{
    // Initial level: L
    Plaintext plain;
    Ciphertext cipher,tmp_cipher;
    // Compute Cheby basis
    vector<Ciphertext> cheby_basis;
    build_cheby_basis(input,cheby_basis,4,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    printf("BUILT CHEBY BASIS\n");
    // Compute first level.
    double coeff_fq1 = -1.121704102;
    double coeff_fr1 =  1.978370667;

    //decrypt_and_print_and_max_round(input,decryptor,encoder,1.0,0,5,5);
    output = input;
    evaluator.multiply_const(input,coeff_fq1,output);
    //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0,5,5);
    evaluator.rescale_to_next_inplace(output); // L+1
    //printf("First level complete1: %f %zu %f %zu\n",output.scale(),output.coeff_modulus_size(),cheby_basis[2].scale(),cheby_basis[2].coeff_modulus_size());

    evaluator.multiply_inplace_reduced_error(output,cheby_basis[2],relin_keys);
    evaluator.rescale_to_next_inplace(output); // L+2

    //printf("fq1 done\n");
    //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0,5,5);
    evaluator.multiply_const(input,coeff_fr1,cipher);
    //printf("First level complete1: %f %zu %f %zu\n",output.scale(),output.coeff_modulus_size(),cipher.scale(),cipher.coeff_modulus_size());
    evaluator.rescale_to_next_inplace(cipher);
    

    //printf("First level complete1: %f %zu %f %zu\n",output.scale(),output.coeff_modulus_size(),cipher.scale(),cipher.coeff_modulus_size());
    evaluator.add_inplace_reduced_error(output,cipher); // L+2 
    //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0,5,5);
    //printf("First level complete: %f %zu\n",output.scale(),output.coeff_modulus_size());

    printf("FIRST LEVEL COMPLETE!\n");
    // Compute second level.

    double coeff_frq2_q = -0.6178588867;
    double coeff_frq2_r = 0.403533935;

    // Compute frq2_q
    //decrypt_and_print_and_max_round(cheby_basis[3],decryptor,encoder,1.0,0,5,5);
    evaluator.multiply_const(cheby_basis[3],coeff_frq2_q,cipher);
    evaluator.rescale_to_next_inplace(cipher); // L+2
    printf("frq2 done!\n");

    evaluator.multiply_const(input,coeff_frq2_r,tmp_cipher);//L+1
    //printf("fq2_q_0 done! %f %zu %f %zu\n",cipher.scale(),cipher.coeff_modulus_size(),tmp_cipher.scale(),tmp_cipher.coeff_modulus_size());
    
    evaluator.rescale_to_next_inplace(tmp_cipher);

    //decrypt_and_print_and_max_round(tmp_cipher,decryptor,encoder,1.0,0,5,5);
    //decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0,5,5);
    printf("fq2_q_0 done! %f %zu %f %zu\n",cipher.scale(),cipher.coeff_modulus_size(),tmp_cipher.scale(),tmp_cipher.coeff_modulus_size());
    evaluator.add_inplace_reduced_error(cipher,tmp_cipher); // L+2
    //decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0,5,5);
    //printf("f2q done! %f %zu %f %zu\n",cipher.scale(),cipher.coeff_modulus_size(),cheby_basis[3].scale(),cheby_basis[3].coeff_modulus_size());
    //printf("modulus check 3 %f %zu %f %zu\n",cipher.scale(),cipher.coeff_modulus_size(),cheby_basis[4].scale(),cheby_basis[4].coeff_modulus_size());
    //decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0,5,5);


    //printf("modulus check %zu %zu\n",cipher.coeff_modulus_size(),cheby_basis[3].coeff_modulus_size());
    evaluator.multiply_inplace_reduced_error(cipher,cheby_basis[4],relin_keys); // L+3
    evaluator.rescale_to_next_inplace(cipher); // L+4
    //printf("fq2_q done! %f %zu\n",cipher.scale(),cipher.coeff_modulus_size());
    //decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0,5,5);
    
    //evaluator.mod_switch_to_inplace(output,cipher.parms_id());
    printf("fq3_q done! %f %zu %f %zu\n",output.scale(),output.coeff_modulus_size(),cipher.scale(),cipher.coeff_modulus_size());
    // Compute fr3 = frq2*cheby[3] + fr2
    //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0,5,5);
    evaluator.add_inplace_reduced_error(output,cipher); // L+4
    
    //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0,5,5);
   

    // Compute 3rd and final level
    double coeff_fq3 = 0.3557052612;

    evaluator.multiply_const(input,coeff_fq3,cipher);
    evaluator.rescale_to_next_inplace(cipher);
    //printf("fq4 done! %f %zu %f %zu\n",cheby_basis[5].scale(),cheby_basis[5].coeff_modulus_size(),cipher.scale(),cipher.coeff_modulus_size());

    evaluator.multiply_inplace_reduced_error(cipher,cheby_basis[5],relin_keys);
    evaluator.rescale_to_next_inplace(cipher); // L+6


    //printf("fq5 done! %f %zu %f %zu\n",output.scale(),output.coeff_modulus_size(),cipher.scale(),cipher.coeff_modulus_size());
    
   
    //decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0,5,5);
    //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0,5,5);
    

    evaluator.add_inplace_reduced_error(output,cipher);
    return;
}


void sign_function(TensorCipher &inputs,TensorCipher &outputs, int df,int dg, Bootstrapper &bootstrapper, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys)
{
    printf("COMPUTING G POLY\n");
    // Compute g(g(x))
    Ciphertext cipher,tmp_cipher;
    Plaintext plain;
    vector<double> tmp;
    cipher = inputs.cipher();
    vector<Ciphertext> cheby_basis;
    int i;
    for(i = 0;i < dg/2;i++){
        compute_sign_g(cipher,tmp_cipher,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        compute_sign_g(tmp_cipher,cipher,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    }

    //printf("\nDONE WITH G2\n\n\n");
    if(cipher.coeff_modulus_size() < 8)
        bootstrapper.bootstrap_full_real_3(cipher,cipher);
    printf("COMPUTING F POLY");
    for(i=0;i<df/2;i++){
        compute_sign_f(cipher, tmp_cipher, encoder, encryptor, decryptor, evaluator, gal_keys, relin_keys);
        compute_sign_f(tmp_cipher, cipher, encoder, encryptor, decryptor, evaluator, gal_keys, relin_keys);
    }

    outputs = TensorCipher(cipher);
    
    return;
}

void compute_gelu_p(Ciphertext &input,Ciphertext &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys)
{
    // Initial level: L
    Plaintext plain;
    Ciphertext cipher,tmp_cipher;
    // Compute Cheby basis
    vector<Ciphertext> cheby_basis;
    build_cheby_basis(input,cheby_basis,2,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    
    double q_0 = -0.05745879353,q_1 = -0.005337069175;
    double r_0 = -0.55528939,r_1 = -0.4187418723;

    // Evaluate q
    evaluator.multiply_const(input,q_1,tmp_cipher);
    evaluator.rescale_to_next_inplace(tmp_cipher); // L-1
    evaluator.add_const_inplace(tmp_cipher,q_0); // L-1
    printf("scale: %f,level: %zu\n",tmp_cipher.scale(),tmp_cipher.coeff_modulus_size());
    //decrypt_and_print_and_max_round(tmp_cipher,decryptor,encoder,1.0,0,5,5);
    //decrypt_and_print_and_max_round(cheby_basis[2],decryptor,encoder,1.0,0,5,5);
    evaluator.multiply_inplace_reduced_error(tmp_cipher,cheby_basis[2],relin_keys);
    //decrypt_and_print_and_max_round(tmp_cipher,decryptor,encoder,1.0,0,5,5);

    // Evaluate r
    evaluator.multiply_const(input,r_1,output);
    evaluator.rescale_to_next_inplace(output); // L-1
    evaluator.add_const_inplace(output,r_0);

    printf("tmp scale: %f,level: %zu\n",tmp_cipher.scale(),tmp_cipher.coeff_modulus_size());
    printf("output scale: %f,level: %zu\n",output.scale(),output.coeff_modulus_size());
    evaluator.add_inplace_reduced_error(output,tmp_cipher);

    return;
}
void compute_gelu_q(Ciphertext &input,Ciphertext &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys)
{
     // Initial level: L
     // T0: L
     // T1: L
     // T2: L-2
     // T3: L-4
     // T4: L-4
     // T5: L-6

    Plaintext plain;
    Ciphertext cipher,tmp_cipher;
    // Compute Cheby basis
    vector<Ciphertext> cheby_basis;
    build_cheby_basis(input,cheby_basis,4,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    
    // Evaluate qq2,qr2
    double qq1_0 = 0.1634058825,qq1_1 = -0.00324699876;
    double qr1_0 = 0.1750485092,qr1_1 = 0.5027208006;

    evaluator.multiply_const(input,qq1_1,tmp_cipher);
    evaluator.rescale_to_next_inplace(tmp_cipher);
    evaluator.add_const_inplace(tmp_cipher,qq1_0);
    evaluator.multiply_inplace_reduced_error(tmp_cipher,cheby_basis[2],relin_keys);
    

    evaluator.multiply_const(input,qr1_1,output);
    evaluator.rescale_to_next_inplace(output);
    evaluator.add_const_inplace(output,qr1_0);

    evaluator.add_inplace_reduced_error(output,tmp_cipher);
    

    // Evaluate qq2
    double qq2_0 =  -0.004401064777,qq2_1 = 0.0002609111473, qq_2 = 0.0001533078376;

    // qq2 x^2 + qq1 x + qq0
    evaluator.square(input,tmp_cipher);
    evaluator.relinearize_inplace(tmp_cipher,relin_keys);
    evaluator.rescale_to_next_inplace(tmp_cipher);

    evaluator.multiply_const_inplace(tmp_cipher,qq_2);
    evaluator.rescale_to_next_inplace(tmp_cipher);

    evaluator.multiply_const(input,qq2_1,cipher);
    evaluator.rescale_to_next_inplace(cipher);

    evaluator.add_inplace_reduced_error(cipher,tmp_cipher);

    evaluator.add_const_inplace(cipher,qq2_0);

    // out = qq2 * T4 + qr2
    evaluator.multiply_inplace_reduced_error(cipher,cheby_basis[4],relin_keys);
    
    evaluator.add_inplace_reduced_error(output,cipher);
    return;
}

void gelu(Ciphertext &inputs,Ciphertext &outputs, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys){
                            return;
}

/*
def exp(x,r):
    return math.pow(1+x/(math.pow(2,6)),math.pow(2,6))
*/

void compute_exp(Ciphertext &input,Ciphertext &output,int r, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys)
{
    printf(" START COMPUTING EXP\n");
    double power = pow(2,r);

    // Compute term to exponentiate
    evaluator.multiply_const(input,1/power,output);
    evaluator.rescale_to_next_inplace(output);
    evaluator.add_const_inplace(output,1);
    
    // Exponentiate
    for(int i =0;i<r;i++) {
        evaluator.square_inplace(output);
        evaluator.relinearize_inplace(output,relin_keys);
        evaluator.rescale_to_next_inplace(output);
        printf("SCALE: %f\n",output.scale());
    }
    printf(" END COMPUTING EXP\n");
    return;
}

void compute_softmax(Ciphertext &input,int r,Bootstrapper &bootstrapper, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys)
{   
    int i,j;

    Plaintext plain_ones,plain_zeros;
    Ciphertext cipher,rolled,maxes,exps,summed,inverses;

    vector<double> ones_mask(32768,0.0);
    vector<double> zeros_mask(32768,1.0);

    for(i=0;i<128;i++){
        for(j=0;j<128;j++){
            ones_mask[i*256+128+j] = 1.0;
            zeros_mask[i*256+128+j] = 0.0;
        }
    }

    printf("About to prepare for max computation\n");

    // Prepare cipher for folding
    evaluator.rotate_vector(input,-128,gal_keys,rolled);
    evaluator.add_inplace_reduced_error(input,rolled);

    quickMax(input,maxes,128,bootstrapper,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    printf("MAX RESULT: \n");

    decrypt_and_print_and_max_round(maxes,decryptor,encoder,1.0,0);

    Ciphertext bluh = input;

    evaluator.sub_inplace_reduced_error(input,maxes);

    // Compute exp and subtract out ones
    compute_exp(input,exps,6,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    printf("After Exp\n");
    //encoder.encode(ones_mask,exps.scale(),plain_ones);
    encoder.encode(zeros_mask,exps.scale(),plain_zeros);
    evaluator.mod_switch_to_inplace(plain_zeros,exps.parms_id());
    evaluator.multiply_plain_inplace(exps,plain_zeros);
    //evaluator.sub_plain_inplace(exps,plain_ones);

    // fakeBootstrap(exps,exps,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
     
    bootstrapper.bootstrap_full_real_3(rolled,exps);
    
    // Compute sum
    evaluator.rotate_vector_inplace(rolled,-128,gal_keys);
    evaluator.add_inplace_reduced_error(rolled,exps);
    quickSum(rolled,summed,128,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    printf("After QuickSum\n");
    decrypt_and_print_and_max_round(summed,decryptor,encoder,1.0,0);


    Ciphertext cipher_in,cipher_out;
    Plaintext plain;

    // fakeBootstrap(summed,cipher_out,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    // Goldschmidt division
    compute_inverse(summed,inverses,4,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    evaluator.multiply_reduced_error(exps,inverses,relin_keys,input);
    evaluator.rescale_to_next_inplace(input);
}

void compute_smax(Ciphertext &input,int r, int gamma, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys)
{   
    int i,j;

    Plaintext plain_ones,plain_zeros,plain_gamma;
    Ciphertext cipher,rolled,maxes,exps,summed,inverses;

    vector<double> ones_mask(32768,0.0);
    vector<double> zeros_mask(32768,1.0);
    vector<double> gamma_mask(32768,0.0);

    for(i=0;i<128;i++){
        for(j=0;j<128;j++){
            ones_mask[i*256+128+j] = 1.0;
            zeros_mask[i*256+128+j] = 0.0;
            gamma_mask[i*256+128+j] = -gamma;
        }
    }

    Ciphertext bluh = input;

    encoder.encode(gamma_mask,input.scale(),plain_gamma);
    evaluator.mod_switch_to_inplace(plain_gamma,input.parms_id());
    printf("About to prepare for max computation\n");

    evaluator.add_plain_inplace(input,plain_gamma);

    // Compute exp and subtract out ones
    compute_exp(input,exps,6,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    printf("After Exp\n");
    //encoder.encode(ones_mask,exps.scale(),plain_ones);
    encoder.encode(zeros_mask,exps.scale(),plain_zeros);
    evaluator.mod_switch_to_inplace(plain_zeros,exps.parms_id());
    evaluator.multiply_plain_inplace(exps,plain_zeros);
    //evaluator.sub_plain_inplace(exps,plain_ones);

    // fakeBootstrap(exps,exps,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    
    // Compute sum
    evaluator.rotate_vector(exps,-128,gal_keys,rolled);
    evaluator.add_inplace_reduced_error(rolled,exps);
    quickSum(rolled,summed,128,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    printf("After QuickSum\n");
    decrypt_and_print_and_max_round(summed,decryptor,encoder,1.0,0);


    Ciphertext cipher_in,cipher_out;
    Plaintext plain;
    // fakeBootstrap(summed,cipher_out,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    // Goldschmidt division
    compute_inverse(summed,inverses,4,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    evaluator.multiply_reduced_error(exps,inverses,relin_keys,input);
    evaluator.rescale_to_next_inplace(input);
}
