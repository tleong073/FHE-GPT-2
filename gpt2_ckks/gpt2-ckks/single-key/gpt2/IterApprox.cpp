#include "gpt2_seal.h"



/*

def goldschmidt_division(n,d,iters):
    for _ in range(iters):
        n = 2*n-n*d
        d = 2*d-d*d
    return n
*/

// Assume 
void compute_inverse(Ciphertext &input,Ciphertext &output,int iters, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys)
{
    Ciphertext two_cipher,d_cipher,f_cipher;
    Plaintext plain;
    encoder.encode(1.0,input.scale(),plain);
    encryptor.encrypt(plain,output);

    encoder.encode(2.0,input.scale(),plain);
    encryptor.encrypt(plain,two_cipher);

    d_cipher = input;
    for(int i = 0 ; i<iters;i++) {
        printf("start\n",i);
        // f = 2-d
        evaluator.sub_reduced_error(two_cipher,d_cipher,f_cipher);
        evaluator.multiply_const_inplace(two_cipher,1.0);
        evaluator.rescale_to_next_inplace(two_cipher);

        // n = n * f
        evaluator.multiply_inplace_reduced_error(output,f_cipher,relin_keys);
        evaluator.rescale_to_next_inplace(output);
        // d = d*f
        evaluator.multiply_inplace_reduced_error(d_cipher,f_cipher,relin_keys);
        evaluator.rescale_to_next_inplace(d_cipher);

    }
    return;
}

/*
def newton_iteration(a,iters):
    x = 0.1
    for i in range(iters):
        x = x*(1.5-0.5*a*(x**2))
    return x
*/
void compute_inv_sqrt(Ciphertext &input,Ciphertext &output,int iters,double guess, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys)
{
    Ciphertext cipher,const_cipher,tmp_cipher;
    Plaintext plain;
    encoder.encode(guess,input.scale(),plain);
    encryptor.encrypt(plain,output);


    printf("start\n");
    for(int i = 0 ; i<iters;i++) {

        evaluator.multiply_const(input,-0.5,tmp_cipher);
        evaluator.rescale_to_next_inplace(tmp_cipher);

        evaluator.square(output,cipher);
        evaluator.relinearize_inplace(cipher,relin_keys);
        evaluator.rescale_to_next_inplace(cipher);

        evaluator.multiply_inplace_reduced_error(cipher,tmp_cipher,relin_keys);
        evaluator.rescale_to_next_inplace(cipher);
        
        evaluator.add_const_inplace(cipher,1.5);

        evaluator.multiply_inplace_reduced_error(output,cipher,relin_keys);
        evaluator.rescale_to_next_inplace(output);

        
        decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0,5,5);
    }
    return;
}

/*
def exp(x,r):
    return math.pow(1+x/(math.pow(2,6)),math.pow(2,6))
*/

void compute_exp(Ciphertext &input,Ciphertext &output,int r, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys)
{
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

    return;
}