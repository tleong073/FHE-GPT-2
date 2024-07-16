#ifndef NETWORK_UTIL_H
#define NETWORK_UTIL_H

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

using grpc::ClientContext;
using grpc::ClientReaderWriter;
using grpc::Channel;
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
using secureinference::Ciphertexts;
using secureinference::Plaintexts;
using secureinference::ByteData;
using secureinference::InitResponse;
using secureinference::SecureInference;
using std::chrono::system_clock;

using namespace seal;
using namespace std;

#define BLAH 12

void add_two();

void unpack_cipher(SEALContext seal_context,const secureinference::Ciphertexts *ciphers, vector<seal::Ciphertext> &ciphertexts);

void unpack_plain(SEALContext seal_context,const secureinference::Plaintexts *plains, std::vector<seal::Plaintext> &plaintexts);

void pack_cipher(SEALContext seal_context,vector<seal::Ciphertext> ciphertexts,secureinference::Ciphertexts *ciphers);

void pack_plain(SEALContext seal_context,vector<seal::Plaintext> plaintexts,secureinference::Plaintexts *plains);

#endif //NETWORK_UTIL_H
