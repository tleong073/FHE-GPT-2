
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <random>

// Seal includes
#include "seal/seal.h"

// grpc includes
#include <grpc/grpc.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include "secure_inference.grpc.pb.h"

#include "header/network_util.h"

using namespace seal;

class SecureInferenceClient {
 public:
  SecureInferenceClient(std::shared_ptr<Channel> channel)
      : stub_(secureinference::SecureInference::NewStub(channel)) {}
  
 void InitServer() {
    parms_ = EncryptionParameters(scheme_type::ckks);
    secureinference::Params params;
    secureinference::InitResponse resp;
    ClientContext context;

    std::stringstream parms_stream;
    std::stringstream rlk_stream;
    std::stringstream gal_stream; 


    size_t poly_modulus_degree = 65536;
    parms_.set_poly_modulus_degree(poly_modulus_degree);
    parms_.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 50, 30,30,50 }));

    
    params.set_parms_size(parms_.save(parms_stream));

    SEALContext seal_context(parms_,true,sec_level_type::none);
    KeyGenerator keygen(seal_context);
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

  void SendInput(int num_inputs) {
    secureinference::Plaintexts plains;
    secureinference::InitResponse resp;
    ClientContext context;

    SEALContext seal_context(parms_,true,sec_level_type::none);
    vector<Plaintext> plaintexts;
    CKKSEncoder encoder(seal_context);
    Evaluator evaluator(seal_context);
    Encryptor encryptor(seal_context,pk_);
    Decryptor decryptor(seal_context,sk_);

    printf("Inside send input. Initialization complete\n");

    Plaintext plain;
    size_t scale = pow(2,30);
    int i,j;
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> unif;
    vector<double> vec(32768);
    
    vector<double> vec2(32768);

    // TODO: Make meaningful inputs for correctness
    for(i=0;i<num_inputs;i++){
      encoder.encode(i,scale,plain);
      plain.set_zero();
      
      std::generate(vec.begin(), vec.end(), [&](){ return unif(gen); });
      std::generate(vec2.begin(), vec2.end(), [&](){ return unif(gen); });

      // Subtract out random poly
      for(j=0;j<8192;j++){
        plain[j] = vec[j] - vec2[j];
      }

      plaintexts.push_back(plain);
    }

    pack_plain(seal_context,plaintexts,&plains);

    Status status = stub_->SendInput(&context,plains,&resp);

    if(status.ok()){
       std::cout << " SendInput rpc succeeded." << std::endl;
    } else {
       std::cout << "SendInput rpc failed." << status.error_message()  << std::endl;
    }
  }

  void repack_responses(vector<vector<Plaintext>>& response,vector<seal::Plaintext> &input,int offset){
    int i,j,k,row,col;
    for(i=0;i<3;i++){
      for(j=0;j<128;j++){
        row=offset;
        for(k=0;k<256;k++){
          col=i*256+k;
          response[k/64][k % 64] = input[0][j];
        }
      }
    }
  }

  void repack() {
    ClientContext context;
    chrono::time_point deadline = std::chrono::system_clock::now() +
    std::chrono::seconds(100000);
    context.set_deadline(deadline);

    std::shared_ptr<ClientReaderWriter<secureinference::Plaintexts, secureinference::Ciphertexts> > stream(
        stub_->RepackAndSend(&context));


    SEALContext seal_context(parms_,true,sec_level_type::none);
    CKKSEncoder encoder(seal_context);
    Evaluator evaluator(seal_context);
    Encryptor encryptor(seal_context,pk_);
    Decryptor decryptor(seal_context,sk_);
    size_t scale = pow(2,30);

    Plaintext plain;
    Ciphertext cipher;
    vector<Plaintext> plaintexts;
    vector<Plaintext> tmp_plaintexts;
    vector<vector<Plaintext>> resp_plaintexts;
    vector<seal::Ciphertext> ciphertexts;

    vector<secureinference::Plaintexts> bulk_plains;
    secureinference::Plaintexts plains;
    secureinference::Ciphertexts ciphers;

    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> unif;

    vector<double> vec(32768);

    std::generate(vec.begin(), vec.end(), [&](){ return unif(gen); });

    cout << BLAH << endl;


    std::stringstream ss_in;
    encoder.encode(vec,scale,plain);
    encryptor.encrypt(plain,cipher);
    cout << "Before writing: " <<endl;
    cipher.save(ss_in);

    secureinference::ByteData* b_data = plains.add_plain();
    b_data->set_value(ss_in.str());
    

    bool k = stream->Write(plains);
    cout << "After writing: " << k << endl;

    plains.clear_plain();

    plaintexts.clear();
    int i,j;
    plain.set_zero();
    for(i = 0; i<12;i++){
      tmp_plaintexts.clear();
      for(j = 0;j<64;j++){
        tmp_plaintexts.push_back(plain);
      }
      resp_plaintexts.push_back(tmp_plaintexts);
    }

    uint32_t sz;
    int offset=0;
    cout << "READING " <<endl;
    while(offset < 128){
      stream->Read(&ciphers);
      cout << "inside readings " << ciphers.ciphers_size() <<endl;
      ciphertexts.clear();
      plaintexts.clear();
      unpack_cipher(seal_context,&ciphers,ciphertexts);
      for(auto cipher : ciphertexts){
        decryptor.decrypt(cipher,plain);
        plaintexts.push_back(plain);
      }
      repack_responses(resp_plaintexts,plaintexts,offset++);
    }
    cout << "AFTER READING " << stream->NextMessageSize(&sz) <<endl;
    for(i=0;i<12;i++){
      plains.clear_plain();
      pack_plain(seal_context,resp_plaintexts[i],&plains);
      bulk_plains.push_back(plains);
    }

    for (const secureinference::Plaintexts &p : bulk_plains) {
      cout << stream->Write(plains) << endl;
    }
    stream->WritesDone();

    Status status = stream->Finish();

    if (!status.ok()) {
      std::cout << "Repack rpc failed." << status.error_message()  <<std::endl;
    }

  }

    



  /*
  void RouteChat() {
    ClientContext context;

    std::shared_ptr<ClientReaderWriter<RouteNote, RouteNote> > stream(
        stub_->RouteChat(&context));

    std::thread writer([stream]() {
      std::vector<RouteNote> notes{MakeRouteNote("First message", 0, 0),
                                   MakeRouteNote("Second message", 0, 1),
                                   MakeRouteNote("Third message", 1, 0),
                                   MakeRouteNote("Fourth message", 0, 0)};
      for (const RouteNote& note : notes) {
        std::cout << "Sending message " << note.message() << " at "
                  << note.location().latitude() << ", "
                  << note.location().longitude() << std::endl;
        stream->Write(note);
      }
      stream->WritesDone();
    });

    RouteNote server_note;
    while (stream->Read(&server_note)) {
      std::cout << "Got message " << server_note.message() << " at "
                << server_note.location().latitude() << ", "
                << server_note.location().longitude() << std::endl;
    }
    writer.join();
    Status status = stream->Finish();
    if (!status.ok()) {
      std::cout << "RouteChat rpc failed." << std::endl;
    }
  }
  /*
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
  */
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
    //client.TestEval();

    auto start = chrono::high_resolution_clock::now();
    //client.SendInput(4608);
    client.repack();
    auto duration = chrono::high_resolution_clock::now() - start;
    cout << " Took: " << duration.count() / 1000 / 1000<< "ms" << endl;
    
    return 0;
}