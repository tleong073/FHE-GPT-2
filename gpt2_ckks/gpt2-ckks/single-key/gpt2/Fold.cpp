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

void quickSum(TensorCipher &input,TensorCipher &output,int n, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys) {
    
    Ciphertext cipher = input.cipher(),rot_cipher;
    evaluator.rotate_vector(cipher,-n,gal_keys,rot_cipher);
    evaluator.add_inplace_reduced_error(cipher,rot_cipher);
    int acc = 1;
    for(int i = 0; i<log(n);i++) {
        decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0);
        evaluator.rotate_vector(cipher,acc,gal_keys,rot_cipher);
        evaluator.add_inplace_reduced_error(cipher,rot_cipher);
        acc *= 2;
    }
    output.set_ciphertext(cipher);
    return;
}

void computeMax(Ciphertext &input1,Ciphertext &input2,Ciphertext &output, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys) {
    Ciphertext cipher,diff_cipher,sign_cipher;
    evaluator.sub(input1,input2,diff_cipher);

    TensorCipher t = TensorCipher(diff_cipher);
    TensorCipher out;

    sign_function(t,out,2,2,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    sign_cipher = out.cipher();
    fakeBootstrap(sign_cipher,sign_cipher,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    evaluator.multiply_inplace_reduced_error(diff_cipher,sign_cipher,relin_keys);
    evaluator.rescale_to_next_inplace(diff_cipher);

    evaluator.add_inplace_reduced_error(diff_cipher,input1);
    evaluator.add_inplace_reduced_error(diff_cipher,input2);

    evaluator.multiply_const(diff_cipher,0.5,output);
    evaluator.rescale_to_next_inplace(output);

    return;
}

void quickMax(TensorCipher &input,TensorCipher &output,int n, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys) {
    Ciphertext cipher = input.cipher(),rot_cipher;

    evaluator.rotate_vector(cipher,-n,gal_keys,rot_cipher);
    evaluator.add_inplace(cipher,rot_cipher);

    int acc = 1;
    for(int i = 0; i<log(n);i++) {
        evaluator.rotate_vector(cipher,acc,gal_keys,rot_cipher);
        computeMax(cipher,rot_cipher,cipher,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        fakeBootstrap(cipher,cipher,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        acc *= 2;
    }
    output.set_ciphertext(cipher);
    return;
}