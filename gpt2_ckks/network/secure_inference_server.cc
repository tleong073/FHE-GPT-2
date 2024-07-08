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


using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;
using secureinference::Params;
using secureinference::Query;
using secureinference::Response;
using secureinference::Ciphertext;
using secureinference::InitResponse;
using secureinference::SecureInference;
using std::chrono::system_clock;

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

    SEALContext seal_context(parms_);

    
    relin_keys_.load(seal_context,rlk_stream);
    gal_keys_.load(seal_context,gal_stream);

    response->set_status("Good");

    std::cout << "Done with server init \n";

    return Status::OK;
  }

  Status TestEval(ServerContext *context, const Query* query, Response *res){
    std::cout << "Inside TestEval!" << std::endl;

    secureinference::Ciphertext *out_cipher = res->add_ciphers();

    SEALContext seal_context(parms_);
    Evaluator evaluator(seal_context);

    seal::Ciphertext ctxt;
    std::stringstream ss_in;
    std::stringstream ss_out;

    std::cout << "Inside TestEval! " << query->ciphers_size() << " " << std::endl;
    for(secureinference::Ciphertext cipher : query->ciphers()) {
      ss_in.str(cipher.value());
      ctxt.load(seal_context,ss_in);

      evaluator.rotate_vector_inplace(ctxt,1,gal_keys_);

      out_cipher->set_size(ctxt.save(ss_out));
      out_cipher->set_value(ss_out.str());
      
      res->mutable_ciphers()->AddAllocated(out_cipher);
    }
    

    return Status::OK;
  }

  Status EmbedQuery(ServerContext *context, const Query* query, Response *res){
    // TODO: Implement
    return Status::OK;
  }

 private:
  EncryptionParameters parms_;
  RelinKeys relin_keys_;
  GaloisKeys gal_keys_;
};


void RunServer() {
  std::string server_address("0.0.0.0:50051");
  SecureInferenceImpl service;


  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.SetMaxReceiveMessageSize(1024*1024 * 60);
  builder.SetMaxSendMessageSize(1024*1024 * 60);
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