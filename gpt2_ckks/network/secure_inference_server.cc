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
#include "secure_inference.grpc.pb.h"

#include "header/network_util.h"

using namespace seal;


class SecureInferenceImpl final : public SecureInference::Service {
 public:
  explicit SecureInferenceImpl() {}

  Status InitServer(ServerContext *context, const Params* params, InitResponse *response) override {  
    std::cout << "Inside InitServer \n";
    if(params->parms_size() <= 0)
        return Status::CANCELLED;
    
    std::stringstream parms_stream(params->parms());
    std::stringstream rlk_stream(params->relin_keys());
    std::stringstream gal_stream(params->gal_keys());

    parms_.load(parms_stream);

    SEALContext seal_context(parms_,true,sec_level_type::none);

    
    relin_keys_.load(seal_context,rlk_stream);
    gal_keys_.load(seal_context,gal_stream);

    response->set_status("Good");

    std::cout << "Done with server init \n";

    return Status::OK;
  }

  /*
  Status TestEval(ServerContext *context, const Query* query, Response *res){
    std::cout << "Inside TestEval!" << std::endl;

    

    SEALContext seal_context(parms_);
    Evaluator evaluator(seal_context);

    seal::Ciphertext ctxt;
    std::stringstream ss_in;
    std::stringstream ss_out;

    std::cout << "Inside TestEval! " << query->ciphers_size() << " " << std::endl;
    for(secureinference::Ciphertext cipher : query->ciphers()) {
      secureinference::Ciphertext *out_cipher = res->add_ciphers();

      ss_in.str(cipher.value());
      ctxt.load(seal_context,ss_in);

      evaluator.rotate_vector_inplace(ctxt,1,gal_keys_);

      out_cipher->set_size(ctxt.save(ss_out));
      out_cipher->set_value(ss_out.str());
    }
    

    return Status::OK;
  }
  */
  
  Status SendInput(ServerContext* context, const secureinference::Plaintexts *plains,secureinference::InitResponse *resp) override{
    
    SEALContext seal_context(parms_);
    
    vector<Plaintext> plaintexts;
    
    unpack_plain(seal_context,plains,plaintexts);
    
    return Status::OK;
  }

  Status RepackAndSend(ServerContext* context,
                  ServerReaderWriter<secureinference::Ciphertexts,secureinference::Plaintexts>* stream) override {
  /*
  RouteNote note;
  while (stream->Read(&note)) {
    std::unique_lock<std::mutex> lock(mu_);
    for (const RouteNote& n : received_notes_) {
      if (n.location().latitude() == note.location().latitude() &&
          n.location().longitude() == note.location().longitude()) {
        stream->Write(n);
      }
    }
    received_notes_.push_back(note);
  }
  */

  SEALContext seal_context(parms_,true,sec_level_type::none);

  Plaintext plain;
  vector<Plaintext> plaintexts;
  vector<seal::Ciphertext> ciphertexts;
  // Initial read should be empty
  secureinference::Plaintexts plains;
  

  cout << "READING CLIENT INPUT" << endl; 
  cout << stream->Read(&plains) << endl;
  // TODO: Fix misleading typing later
  seal::Ciphertext tmp_cipher;
  std::stringstream ss_in,ss_out;

  cout << "LOADING CIPHERTEXT" << endl; 

  // Receive dummy cipher to send back
  for(auto cipher : plains.plain()) {
    ss_in.str(cipher.value());
    tmp_cipher.load(seal_context,ss_in);
    ciphertexts.push_back(tmp_cipher);
  }

  cout << "SENDING BACK CIPHERS"  << endl;

  // Assume computations are done in-between.

  ciphertexts[0].save(ss_out);

  
  /*
  tmp_byte_data = ciphers.add_ciphers();
  tmp_byte_data->set_value(ss_out.str());
  tmp_byte_data->set_size(ss_out.str().size());

  tmp_byte_data = ciphers.add_ciphers();
  tmp_byte_data->set_value(ss_out.str());
  tmp_byte_data->set_size(ss_out.str().size());
  */

  // Send back ciphers
  int i;
  std::string s = ss_out.str();
  std::string s2 = "Hello";
  for(i=0;i<768;i++){
    secureinference::Ciphertexts ciphers;
    secureinference::ByteData* tmp_byte_data = ciphers.add_ciphers();
    tmp_byte_data->set_value(s);
    tmp_byte_data->set_size(ss_out.str().size());
    cout <<ss_out.str().size() << endl;
    //cout <<ss_out.str().size() << endl;
    

    bool a = stream->Write(ciphers);
    cout << "WROTE CIPHER: " << i << " " << a <<  endl;
  }
  
  cout << "ABOUT TO READ: " <<  endl;
  while (stream->Read(&plains)) {
    cout << "INSIDE server read" << endl;
    unpack_plain(seal_context,&plains,plaintexts);
  }

  cout << "Done: " <<  endl;

  return Status::OK;
}

  /*
  Status EmbedQuery(ServerContext *context, const Query* query, Response *res){
    // TODO: Implement
    return Status::OK;
  }
  */
  

 private:
  EncryptionParameters parms_;
  RelinKeys relin_keys_;
  GaloisKeys gal_keys_;
  std::mutex mu_;
};


void RunServer() {
  std::string server_address("0.0.0.0:50051");
  SecureInferenceImpl service;


  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.SetMaxReceiveMessageSize(1024*1024 * 1420);
  builder.SetMaxSendMessageSize(1024*1024 * 1420);
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

int main(int argc, char** argv) {
  // Expect only arg: --db_path=path/to/route_guide_db.json.
  RunServer();
  return 0;
}