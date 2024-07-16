#include "approx.h"


/*
def quickMax(vec,n):
    for i in range(1,int(math.log(n,2))+1):
        rot_vec =  np.roll(vec,-i)
        tmp_vec = np.column_stack((vec,rot_vec))
        vec = np.apply_along_axis(lambda x: max(x[0],x[1]),1,tmp_vec)

    return vec

def quickSum(vec,n):
    for i in range(1,int(math.log(n,2))+1):
        rot_vec =  np.roll(vec,-i)
        vec = vec + rot_vec
    return vec
*/

void quickSum(Ciphertext &input,Ciphertext &output,int n, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys) 
{
    
    Ciphertext cipher;
    cipher.reserve(16);
    int acc = 1;
    evaluator.rotate_vector(input,acc,gal_keys,cipher);
    evaluator.add(input,cipher,output);
    acc *= 2;

    //printf("Output \n");
    //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0);
    //printf("-----GAP-----\n");
    for(int i = 0; i<log2(n)-1;i++) {
        evaluator.rotate_vector(output,acc,gal_keys,cipher);
        //printf("Cipher \n");
        //decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0);
        evaluator.add_inplace_reduced_error(output,cipher);
        //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0);
        acc *= 2;
    }
    //printf("Done with quick sum\n");
    return;
}

// Computes max = 0.5*((a+b)*((a-b)*sign(a-b)))
void computeMax(Ciphertext &input1,Ciphertext &input2,Ciphertext &output, Bootstrapper &bootstrapper, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys) {
    
    printf("COMPUTING MAX\n");
    Ciphertext cipher,diff_cipher,sign_cipher,normalized_diff;
    evaluator.sub(input1,input2,diff_cipher);
    printf("Done WITH SUB\n");
    evaluator.multiply_const(diff_cipher,0.1,normalized_diff);
    evaluator.rescale_to_next_inplace(normalized_diff);

    TensorCipher t = TensorCipher(normalized_diff);
    TensorCipher out;
    printf("COMPUTING SIGN: %zu\n",normalized_diff.coeff_modulus_size());
    sign_function(t,out,2,2,bootstrapper,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    sign_cipher = out.cipher();
    printf("Sign Cipher: %f %zu \n",sign_cipher.scale(),sign_cipher.coeff_modulus_size());
    //decrypt_and_print_and_max_round(sign_cipher,decryptor,encoder,1.0,0);
    //decrypt_and_print_and_max_round(diff_cipher,decryptor,encoder,1.0,0);

    evaluator.multiply_inplace_reduced_error(diff_cipher,sign_cipher,relin_keys);
    evaluator.rescale_to_next_inplace(diff_cipher);

    //decrypt_and_print_and_max_round(diff_cipher,decryptor,encoder,1.0,0);

    evaluator.add_inplace_reduced_error(diff_cipher,input1);
    evaluator.add_inplace_reduced_error(diff_cipher,input2);

    //decrypt_and_print_and_max_round(diff_cipher,decryptor,encoder,1.0,0);
    evaluator.multiply_const(diff_cipher,0.5,output);
    evaluator.rescale_to_next_inplace(output);
    //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0);

    printf("Sign ouput parameter levels: %f %zu\n",output.scale(),output.coeff_modulus_size());

    return;
}

// Assume input is formatted for fold
void quickMax(Ciphertext &input,Ciphertext &output,int n, Bootstrapper &bootstrapper,CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys) {
    Ciphertext cipher = input,rot_cipher,tmp_cipher,res_cipher;

    int acc = 1;
    for(int i = 0; i<log2(n);i++) {
        tmp_cipher=cipher;    
        evaluator.rotate_vector(tmp_cipher,acc,gal_keys,rot_cipher);
        printf("QuickMax iteration: %d\n",i);
        computeMax(tmp_cipher,rot_cipher,cipher,bootstrapper,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        if(cipher.coeff_modulus_size() < 18)
            bootstrap(cipher,cipher,bootstrapper,evaluator);
        printf("DONE WITH MAX!\n");
        acc *= 2;
    }
    output = cipher;
    return;
}