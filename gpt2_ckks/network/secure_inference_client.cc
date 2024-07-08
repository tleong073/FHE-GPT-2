
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

// Seal includes
#include "seal/seal.h"

// grpc includes
#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/create_channel.h>
#include "secure_inference.grpc.pb.h"


using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;
using secureinference::Params;
using secureinference::Query;
using secureinference::Response;
using secureinference::Ciphertext;
using secureinference::SecureInference;
using std::chrono::system_clock;

using namespace seal;

class SecureInferenceClient {
 public:
  SecureInferenceClient(std::shared_ptr<Channel> channel)
      : stub_(SecureInference::NewStub(channel)) {}
  
 void InitServer() {
    parms_ = EncryptionParameters(scheme_type::ckks);
    secureinference::Params params;
    secureinference::InitResponse resp;
    ClientContext context;

    std::stringstream parms_stream;
    std::stringstream rlk_stream;
    std::stringstream gal_stream; 


    size_t poly_modulus_degree = 8192;
    parms_.set_poly_modulus_degree(poly_modulus_degree);
    parms_.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 50, 30,30,50 }));

    
    params.set_parms_size(parms_.save(parms_stream));

    SEALContext seal_context(parms_);
    KeyGenerator keygen(seal_context);
    printf("HERE4\n");
    sk_ = keygen.secret_key();
    keygen.create_public_key(pk_);
    keygen.create_relin_keys(relin_keys_);
    keygen.create_galois_keys(gal_keys_);

    params.set_relin_key_size(relin_keys_.save(rlk_stream));
    params.set_gal_key_size(gal_keys_.save(gal_stream));

    params.set_parms(parms_stream.str());
    params.set_relin_keys(rlk_stream.str());
    params.set_gal_keys(gal_stream.str());


    Status status = stub_->InitServer(&context,params,&resp);

    if (status.ok()) {
      std::cout << " InitServer rpc succeeded." << std::endl;
    } else {
      std::cout << "InitServer rpc failed." << status.error_message()  << std::endl;
    }
  }

  void TestEval() {
    secureinference::Query query;
    secureinference::Response resp;
    ClientContext context;

    std::stringstream parms_stream;
    std::stringstream data_stream;



    SEALContext seal_context(parms_);
    KeyGenerator keygen(seal_context);

    CKKSEncoder encoder(seal_context);
    Evaluator evaluator(seal_context);
    Encryptor encryptor(seal_context,pk_);
    Decryptor decryptor(seal_context,sk_);
    std::vector<double> v = {1,2,3,4,5};
    Plaintext plain;


    secureinference::Ciphertext *serial_cipher = query.add_ciphers();

    encoder.encode(v,pow(2.0,30),plain);
    serial_cipher->set_size(encryptor.encrypt(plain).save(data_stream));
    serial_cipher->set_value(data_stream.str());


    Status status = stub_->TestEval(&context,query,&resp);

    if (status.ok()) {
      std::cout << " TestEval rpc succeeded." << std::endl;
    } else {
      std::cout << "TestEval rpc failed." << status.error_message()  << std::endl;
      return;
    }

    seal::Ciphertext cipher;
    data_stream.str(resp.ciphers().at(0).value());

    std::vector<double> res;
    cipher.load(seal_context,data_stream);
    decryptor.decrypt(cipher,plain);
    encoder.decode(plain,res);

    for(int i = 0;i<5;i++)
      std::cout << res[i] << " ";
    
    std::cout << std::endl;

  }

  const float kCoordFactor_ = 10000000.0;
  std::unique_ptr<SecureInference::Stub> stub_;
  EncryptionParameters parms_;
  RelinKeys relin_keys_;
  SecretKey sk_;
  PublicKey pk_;
  GaloisKeys gal_keys_;
};

int main(int argc, char** argv) {
    SecureInferenceClient client(
    grpc::CreateChannel("localhost:50051",
                        grpc::InsecureChannelCredentials()));
    client.InitServer();
    client.TestEval();
    return 0;
}