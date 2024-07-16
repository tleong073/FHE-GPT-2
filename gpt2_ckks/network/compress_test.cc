
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

// Seal includes
#include "seal/seal.h"
#include "seal/util/polyarithsmallmod.h"

#include <omp.h>

using namespace seal;
using namespace std;


void print_plain(Decryptor &decryptor, const Ciphertext &cipher,size_t scale) {
    Plaintext plain;
    decryptor.decrypt(cipher,plain);
    for(int i = 0; i< 10;i++){
        cout << (Plaintext::pt_coeff_type)plain[i] / pow(2,scale) << " ";
    }
    cout << endl;
}

inline void multiply_power_of_X(const Ciphertext &encrypted,
                                           Ciphertext &destination,
                                           uint32_t index,
                                           EncryptionParameters enc_params) {

  auto coeff_mod_count = enc_params.coeff_modulus().size() - 1;
  auto coeff_count = enc_params.poly_modulus_degree();
  auto encrypted_count = encrypted.size();

  // cout << "coeff mod count for power of X = " << coeff_mod_count << endl;
  // cout << "coeff count for power of X = " << coeff_count << endl;

  // First copy over.
  destination = encrypted;

  // Prepare for destination
  // Multiply X^index for each ciphertext polynomial
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < encrypted_count; i++) {
    for (int j = 0; j < coeff_mod_count; j++) {
      seal::util::negacyclic_shift_poly_coeffmod(encrypted.data(i) + (j * coeff_count),
                                     coeff_count, index,
                                     enc_params.coeff_modulus()[j],
                                     destination.data(i) + (j * coeff_count));
    }
  }
}

vector<Ciphertext> expand_query(const Ciphertext &encrypted,
                                                  uint32_t m,
                                                  EncryptionParameters enc_params,
                                                  GaloisKeys &gal_keys,
                                                  Evaluator &evaluator,
                                                  Decryptor &decryptor) {

  GaloisKeys &galkey = gal_keys;

  // Assume that m is a power of 2. If not, round it to the next power of 2.
  uint32_t logm = ceil(log2(m));
  Plaintext two("2");

  vector<int> galois_elts;
  auto n = enc_params.poly_modulus_degree();
  if (logm > ceil(log2(n))) {
    throw logic_error("m > n is not allowed.");
  }
  for (int i = 0; i < ceil(log2(n)); i++) {
    galois_elts.push_back((n + pow(2, i)) /
                          pow(2, i));
  }

  vector<Ciphertext> temp;
  temp.push_back(encrypted);
  Ciphertext tempctxt;
  Ciphertext tempctxt_rotated;
  Ciphertext tempctxt_shifted;
  Ciphertext tempctxt_rotatedshifted;
  //print_plain(decryptor,encrypted,40);

  for (uint32_t i = 0; i < logm - 1; i++) {
    vector<Ciphertext> newtemp(temp.size() << 1);
    // temp[a] = (j0 = a (mod 2**i) ? ) : Enc(x^{j0 - a}) else Enc(0).  With
    // some scaling....
    int index_raw = (n << 1) - (1 << i);
    int index = (index_raw * galois_elts[i]) % (n << 1);
    //std::cout << "TEMP SIZE " << temp.size() << endl;

    for (uint32_t a = 0; a < temp.size(); a++) {

      evaluator.apply_galois(temp[a], galois_elts[i], galkey,
                               tempctxt_rotated);

      //cout << "shifted " << endl;
      //decryptor.invariant_noise_budget(tempctxt_rotated) << ", ";

      evaluator.add(temp[a], tempctxt_rotated, newtemp[a]);
      multiply_power_of_X(temp[a], tempctxt_shifted, index_raw,enc_params);

      //print_plain(decryptor,tempctxt_shifted,40);

      // cout << "mul by x^pow: " <<
      // client.decryptor_->invariant_noise_budget(tempctxt_shifted) << ", ";

      multiply_power_of_X(tempctxt_rotated, tempctxt_rotatedshifted, index,enc_params);

      // cout << "mul by x^pow: " <<
      // client.decryptor_->invariant_noise_budget(tempctxt_rotatedshifted) <<
      // ", ";

      // Enc(2^i x^j) if j = 0 (mod 2**i).
      evaluator.add(tempctxt_shifted, tempctxt_rotatedshifted,
                      newtemp[a + temp.size()]);
    }
    temp = newtemp;
  }
  // Last step of the loop
  vector<Ciphertext> newtemp(temp.size() << 1);
  int index_raw = (n << 1) - (1 << (logm - 1));
  int index = (index_raw * galois_elts[logm - 1]) % (n << 1);
  for (uint32_t a = 0; a < temp.size(); a++) {
    if (a >= (m - (1 << (logm - 1)))) { // corner case.
      evaluator.multiply_plain(temp[a], two,
                                 newtemp[a]); // plain multiplication by 2.
      // cout << client.decryptor_->invariant_noise_budget(newtemp[a]) << ", ";
    } else {
      evaluator.apply_galois(temp[a], galois_elts[logm - 1], galkey,
                               tempctxt_rotated);
      evaluator.add(temp[a], tempctxt_rotated, newtemp[a]);
      multiply_power_of_X(temp[a], tempctxt_shifted, index_raw,enc_params);
      multiply_power_of_X(tempctxt_rotated, tempctxt_rotatedshifted, index,enc_params);
      evaluator.add(tempctxt_shifted, tempctxt_rotatedshifted,
                      newtemp[a + temp.size()]);
    }
  }

  vector<Ciphertext>::const_iterator first = newtemp.begin();
  vector<Ciphertext>::const_iterator last = newtemp.begin() + m;
  vector<Ciphertext> newVec(first, last);

  return newVec;
}


int main(int argc, char** argv) {

    EncryptionParameters params(scheme_type::ckks);
    GaloisKeys gal_keys;
    RelinKeys relin_keys;
    PublicKey pk;
    SecretKey sk;

    size_t poly_modulus_degree = 32768;
    params.set_poly_modulus_degree(poly_modulus_degree);
    params.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 50, 30,30,50 }));

    SEALContext seal_context(params);
    KeyGenerator keygen(seal_context);
    CKKSEncoder encoder(seal_context);

    keygen.create_public_key(pk);
    sk = keygen.secret_key();

    Encryptor encryptor(seal_context,pk);
    Decryptor decryptor(seal_context,sk);
    Evaluator evaluator(seal_context);

    keygen.create_public_key(pk);
    keygen.create_relin_keys(relin_keys);

    vector<uint32_t> galois_elts;
    int N = params.poly_modulus_degree();
    int logN = log2(N);

    // cout << "printing galois elements...";
    for (int i = 0; i < logN; i++) {
        galois_elts.push_back((N + pow(2, i)) /
                            pow(2, i));
    }
    keygen.create_galois_keys(galois_elts, gal_keys);

    Plaintext plain,plain2;
    Ciphertext cipher;
    vector<double> vec = {1,2,3,4,5,6,7,8,0,0,0};
    vector<double> vec2;

    size_t scale = pow(2.0,30);
    encoder.encode(vec,scale,plain);

    cout <<plain.significant_coeff_count() << endl;
    cout <<plain.coeff_count() << endl;
    cout <<plain.nonzero_coeff_count() << endl;
   
    
    encryptor.set_public_key(pk);
    encoder.decode(plain,vec2);
     int i;
    for(i=0;i<10;i++)
    {
      cout << (Plaintext::pt_coeff_type)plain[i] << " ";
      cout << vec2[i] << " \n";
    }
            
            
    cout << endl;
    encryptor.encrypt(plain,cipher);
    print_plain(decryptor,cipher,1);
    auto start = chrono::high_resolution_clock::now();
   
    vector<Ciphertext> ciphers = expand_query(cipher,128,params,gal_keys,evaluator,decryptor);
    auto duration = chrono::high_resolution_clock::now() - start;
    cout << " Took: " << duration.count() / 1000 / 1000<< "ms" << endl;

    for(auto cipher: ciphers){
        decryptor.decrypt(cipher,plain);
         for(i=0;i<10;i++)
            cout << (Plaintext::pt_coeff_type)plain[i] << " ";
        cout << endl;
    }

    return 0;
}