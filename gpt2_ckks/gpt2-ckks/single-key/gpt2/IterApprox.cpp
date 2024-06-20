#include "gpt2_seal.h"
#include <iomanip>


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
    // Scale to ensure the numbers are  < 1
    double normalize_factor = 0.001;

    vector<double> one_vec(32768,normalize_factor);
    vector<double> two_vec(32768,2.0);

    encoder.encode(one_vec,ENCODE_SCALE,plain);
    evaluator.mod_switch_to_inplace(plain,input.parms_id());
    encryptor.encrypt(plain,output);

    encoder.encode(two_vec,ENCODE_SCALE,plain);
    evaluator.mod_switch_to_inplace(plain,input.parms_id());
    encryptor.encrypt(plain,two_cipher);
    
    
    evaluator.multiply_const(input,normalize_factor,d_cipher);
    evaluator.rescale_to_next_inplace(d_cipher);

    for(int i = 0 ; i<iters;i++) {
        printf("start %d %zu\n",i,output.coeff_modulus_size());
        // f = 2-d
        evaluator.sub_reduced_error(two_cipher,d_cipher,f_cipher);
        printf("After sub \n");

        // n = n * f
        evaluator.multiply_inplace_reduced_error(output,f_cipher,relin_keys);
        evaluator.rescale_to_next_inplace(output);
        printf("After n update: %f %f %zu %zu\n",output.scale(),f_cipher.scale(),output.coeff_modulus_size(),f_cipher.coeff_modulus_size());
        //decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0,5,5);
        
        // d = d*f
        evaluator.multiply_inplace_reduced_error(d_cipher,f_cipher,relin_keys);
        evaluator.rescale_to_next_inplace(d_cipher);
        printf("After d update \n");
    }
    return;
}
/*
# Third order taylor poly
def taylor_expand(x,a):
    coeffs = [(-0.5,-1.5),(-1.5*-0.5,-2.5),(-2.5*-1.5*-0.5,-3.5)]
    out = np.ones_like(x,dtype=complex) *(1/np.sqrt(a-1))
    start_minus_one = np.array(a-1,dtype=complex)
    for i,(coeff,power) in enumerate(coeffs):
        #print(i,out)
        out += coeff * np.power(start_minus_one,power) * np.power((x-a),i+1) * (1/math.factorial(i+1))
        #print(i,out)
    return out
*/
void taylor_expand(Ciphertext &input,Ciphertext &output,int iters,double guess, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys)
{
    printf("start taylor expansion approx\n");

    vector<double> coeffs = {-0.5,-0.5*-1.5,-2.5*-1.5*-0.5};
    vector<double> powers = {-1.5,-2.5,-3.5};

    double coefficient;

    vector<double>a_plus_one(32768,0.0);
    int i,j,fact_acc=1;
    // Always done row-wise so can assume number of rows per cipher
    for(i=0;i<16;i++){
        fill(a_plus_one.begin()+i*(2048),a_plus_one.begin()+(i+1)*(2048),guess+1.0);
    }

    Ciphertext cipher,tmp_cipher = input,tmp2;
    Plaintext plain_ones,plain_a_plus_one,plain;

    encoder.encode(a_plus_one,ENCODE_SCALE,plain_a_plus_one);

    for(i=0;i<3;i++){
        coefficient = coeffs[i]* 1/fact_acc;
        printf("COEFFICIENT: %f %f\n",guess,coefficient);
        cout << std::fixed << std::setprecision(20) << coefficient << endl;
        // Compute power
        evaluator.multiply_const(input,pow(guess,powers[i]/(i+1)),tmp2);
        evaluator.rescale_to_next_inplace(tmp2);
        tmp_cipher = tmp2;
        for(j=0;j<i;j++){
            evaluator.multiply_inplace_reduced_error(tmp_cipher,tmp2,relin_keys);
            evaluator.rescale_to_next_inplace(tmp_cipher);
        }
        evaluator.multiply_const_inplace(tmp_cipher,coefficient);
        evaluator.rescale_to_next_inplace(tmp_cipher);

        // Jank way of ensuring we dont have to initialize using a zero ciphertext
        if(i==0){
            cipher = tmp_cipher;
        } else {
            evaluator.add_inplace_reduced_error(cipher,tmp_cipher);
        }
        // Update accumulated factorial
        fact_acc *= (i+2);

        decrypt_and_print_and_max_round(cipher,decryptor,encoder,1.0,0,5,5);
    }

    output=cipher;
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

    printf("start newton inv sqrt approx\n");
    // Initialize output with taylor expansion guess
    taylor_expand(input,output,3,guess,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0);
    printf("Finish taylor expand\n");
    evaluator.multiply_const(input,-0.5,tmp_cipher);
    evaluator.rescale_to_next_inplace(tmp_cipher);
    
    for(int i = 0 ; i<iters;i++) {

        printf("Finish taylor expand: %d %f %zu\n",i,output.scale(),output.coeff_modulus_size());
        evaluator.square(output,cipher);
        evaluator.relinearize_inplace(cipher,relin_keys);
        evaluator.rescale_to_next_inplace(cipher);
        printf("Finish square: %d %f %zu\n",i,output.scale(),output.coeff_modulus_size());

        evaluator.multiply_inplace_reduced_error(cipher,tmp_cipher,relin_keys);
        evaluator.rescale_to_next_inplace(cipher);
        
        evaluator.add_const_inplace(cipher,1.5);
        printf("Finish cipher process: %d %f %zu\n",i,cipher.scale(),cipher.coeff_modulus_size());

        evaluator.multiply_inplace_reduced_error(output,cipher,relin_keys);
        evaluator.rescale_to_next_inplace(output);

        fakeBootstrap(output,output,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        decrypt_and_print_and_max_round(output,decryptor,encoder,1.0,0,5,5);
    }
    return;
}

void compute_layernorm(Ciphertext &input,Ciphertext &output,vector<double> gamma,vector<double> beta,int row_size, CKKSEncoder &encoder, Encryptor &encryptor, Decryptor &decryptor,
						Evaluator &evaluator, GaloisKeys& gal_keys, RelinKeys &relin_keys)
{

    int rounded_row_size = round_to_2(row_size);
    Plaintext plain_mask,plain_beta;
    Ciphertext cipher,rolled,folded,y,z,inv_sqrt;

    int i;
    for(i=0;i<gamma.size();i++){
        gamma[i] *= sqrt(row_size);
    }

    // Generate zero mask
    vector<double>mask(32768,0.0);
    vector<double>mul_factor(32768,0.0);
    vector<double>beta_factor(32768,0.0);

    printf("LayerNorm: Filling masks:\n");
    for(i=0;i<16;i++){
        fill(mask.begin()+i*(rounded_row_size*2),mask.begin()+i*(rounded_row_size*2)+rounded_row_size,1.0);
        copy(gamma.begin(),gamma.end(),mul_factor.begin()+i*(rounded_row_size*2));
        copy(beta.begin(),beta.end(),beta_factor.begin()+i*(rounded_row_size*2));
    }

    // Compute sum
    evaluator.rotate_vector(input,-rounded_row_size,gal_keys,rolled);
    evaluator.add_inplace_reduced_error(rolled,input);

    quickSum(rolled,folded,rounded_row_size,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    printf("Layernorm Quicksum Complete !: %f %zu\n",folded.scale(),folded.coeff_modulus_size());

    // Multiply by row size and subtract out sum
    evaluator.multiply_const(input,row_size,z);
    evaluator.rescale_to_next_inplace(z);

    decrypt_and_print_and_max_round(z,decryptor,encoder,1.0,0);

    evaluator.sub_inplace_reduced_error(z,folded);

    printf("Layernorm mul by row size complete !: %f %zu\n");

    // Square
    evaluator.square(z,y);
    evaluator.relinearize_inplace(y,relin_keys);
    evaluator.rescale_to_next_inplace(y);

    decrypt_and_print_and_max_round(y,decryptor,encoder,1.0,0);

    printf("Layernorm Square !");

    // Re-format and fold
    evaluator.multiply_vector_inplace_reduced_error(y,mask);
    evaluator.rescale_to_next_inplace(y);


    evaluator.rotate_vector(y,32768-rounded_row_size,gal_keys,rolled);
    evaluator.add_inplace_reduced_error(rolled,y);

    quickSum(rolled,folded,rounded_row_size,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    compute_inv_sqrt(folded,inv_sqrt,4,323251,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    printf("Done computing LAYERNORM inv. square root ! \n");
    fakeBootstrap(inv_sqrt,inv_sqrt,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

    evaluator.multiply_inplace_reduced_error(y,z,relin_keys);
    evaluator.rescale_to_next_inplace(y);

    // Multiply by gamma and sqrt(row size)
    evaluator.multiply_vector_inplace_reduced_error(y,mul_factor);
    evaluator.rescale_to_next_inplace(y);

    // Add beta to finish it off
    encoder.encode(beta,y.scale(),plain_beta);
    evaluator.mod_switch_to_inplace(plain_beta,y.parms_id());
    evaluator.add_plain_inplace(y,plain_beta);
}
/*
def layer_norm(A,gamma,beta,row_size,newton_init_val):

    rounded_row_size = int(round_to_2(row_size))

    # Assume output is packed for bootstrap
    out = np.zeros((3,32768))

    sums = np.zeros_like(A)
    z = np.zeros_like(A)
    y = np.zeros_like(A)
    print(A.shape)

    mask = np.zeros((32768))
    ones = np.pad(np.ones(row_size),(0,32768-row_size),'constant')


    for i in range(16):
        mask += ones
        ones = np.roll(ones,rounded_row_size*2)

    # Assume packed as (128,768) in pre-fold format
    for i in range(A.shape[0]):
        combined = A[i] + np.roll(A[i], rounded_row_size)
        sums[i] = fold.quickSum(combined,rounded_row_size*2)

        if i == 0:
            print(f"folded: {sums[i][:10]} {A[i][:768].sum()}")
        # Compute z = n x 768  - sum
        z[i] = row_size * A[i]
        z[i] = z[i] - sums[i]
        if i == 0:
            print(f"postsqrt EXP: {1/np.sqrt(np.square(z[i][:768]).sum())}")
            print(f"EXP: {math.sqrt(row_size)*z[i][:10]/np.sqrt(np.square(z[i][:768]).sum())}")

        # Compute y = square(z)
        y[i] = z[i]*z[i]

        # Mask out to convert to prefold format
        y[i] = y[i] * mask
        assert (y[i][768:2048] == 0.0).all()

        pre_sum = y[i]
        # Fold again
        y[i] = y[i] + np.roll(y[i],rounded_row_size)
        y[i] = fold.quickSum(y[i],rounded_row_size*2)

        if i == 0:
            print(f"folded2: {y[i][:10]} {pre_sum[:768].sum()}")

        if i == 0:
            print(f"Pre Newton res: {rounded_row_size} {y[i][:10]} {1/np.sqrt(y[i][:10])}")
        
        # Compute inv sqrt w/ Newton method
        y[i] = iter.newton_iteration(y[i]+1,newton_init_val,13)
        #y[i] = np.nan_to_num(y[i],nan=0.0,posinf=0.0,neginf=0.0)
        #y[i] *= mask

        if i == 0:
            print(f"Post Newton res: {y[i][:10]} ")
        # Compute y = z*y
        y[i] = z[i] * y[i]

        y[i] = ((y[i]*gamma) * math.sqrt(row_size)) + beta
        if i == 0:
            print(f"Post norm res: {y[i][:10]} ")
    
    return y#pack.pack_tight(y)
*/