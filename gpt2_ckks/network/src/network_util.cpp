#include "network_util.h"


void add_two(){
    return;
}

void unpack_cipher(SEALContext seal_context,const secureinference::Ciphertexts *ciphers, vector<seal::Ciphertext> &ciphertexts) {
    seal::Ciphertext tmp_cipher;
    std::stringstream ss_in;
    for(auto cipher : ciphers->ciphers()) {
        ss_in.str(cipher.value());
        tmp_cipher.load(seal_context,ss_in);
        ciphertexts.push_back(tmp_cipher);
    }

    return;
}

void unpack_plain(SEALContext seal_context,const secureinference::Plaintexts *plains, std::vector<seal::Plaintext> &plaintexts) {
    seal::Plaintext tmp_plain;
    std::stringstream ss_in;
    for(auto plain: plains->plain()) {
        ss_in.str(plain.value());
        tmp_plain.load(seal_context,ss_in);
        plaintexts.push_back(tmp_plain);
    }

    return;
}

void pack_cipher(SEALContext seal_context,vector<seal::Ciphertext> ciphertexts,secureinference::Ciphertexts *ciphers) {
    secureinference::ByteData *tmp_cipher;
    std::stringstream ss_out;
    for(auto cipher : ciphertexts) {
        tmp_cipher = ciphers->add_ciphers();
        ss_out.clear();
        cipher.save(ss_out);
        tmp_cipher->set_value(ss_out.str());
    }
}

void pack_plain(SEALContext seal_context,vector<seal::Plaintext> plaintexts,secureinference::Plaintexts *plains) {
    secureinference::ByteData *tmp_plain;
    std::stringstream ss_out;
    for(auto cipher : plaintexts) {
        tmp_plain = plains->add_plain();
        ss_out.clear();
        cipher.save(ss_out);
        tmp_plain->set_value(ss_out.str());
    }
}