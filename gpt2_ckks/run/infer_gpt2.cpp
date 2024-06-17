#include "approx.h"
#include "pack.h"

#include "/home/tmleong/FHE-GPT-2/gpt2_weights/weights.pb.h"

#include <chrono>
#include <fstream>
#include <unordered_map>

using namespace std;


string get_key(stringstream *ss,int idx,const char* part1,const char* part2){
    ss->clear();
    *ss << part1 << idx << part2; 
    return ss->str();
}

void load_weight(string key,vector<Ciphertext>* enc_val,vector<vector<double>>* plain_val,gpt2::Weights &weights, CKKSEncoder &encoder, Encryptor &encryptor,bool should_encrypt) {

    int i,j,dim_size,num_ciphers;
    vec tmp;
    tmp.reserve(32768);
    vector<Ciphertext> ciphers;

    Plaintext plain;
	Ciphertext cipher;

    gpt2::Weight weight = weights.value().at(key);

    num_ciphers=1;
    if(weight.dim_size() == 2)
        num_ciphers=weight.dim()[0];
    
    for(i=0;i<num_ciphers;i++){
        auto pt = weight.mutable_plaintexts()->at(i);
        copy(pt.mutable_value()->begin(),pt.mutable_value()->end(),tmp.begin());

        if(should_encrypt){
            encoder.encode(tmp,ENCODE_SCALE,plain);
            encryptor.encrypt(plain,cipher);
            enc_val->push_back(cipher);
        } else {
            plain_val->push_back(tmp);
        }
    }
}

int main() {

    gpt2::Weights weights;
    
    
    // Read weights in from protobuf
    {
        fstream input("/home/tmleong/FHE-GPT-2/gpt2_weights/all_gpt2_weights.pb", ios::in | ios::binary);
        if (!weights.ParseFromIstream(&input)) {
            cerr << "Failed to parse weights." << endl;
            return -1;
        }
    }

    long boundary_K = 25;
	long deg = 59;
    long scale_factor = 2;
    long inverse_deg = 1; 

	long logN = 16;
	long loge = 10;

	long logn = logN-1;
	long logn_2 = 13;
	long logn_3 = 12;
	long sparse_slots = (1 << logn);

	int logp = 50;
	int logq = 51;
	int log_special_prime = 60;

    int log_integer_part = logq - logp - loge + 5;

	// int remaining_level = 14; // Calculation required
	int remaining_level = 21; // Calculation required
	int boot_level = 14; // greater than: subsum 1 + coefftoslot 2 + ModReduction 9 + slottocoeff 2 
	int total_level = remaining_level + boot_level;

    INIT();

    Bootstrapper bootstrapper(loge, logn, logN - 1, total_level, ENCODE_SCALE, boundary_K, deg, scale_factor, inverse_deg, context, keygen, encoder, encryptor, decryptor, evaluator, relin_keys, gal_keys);

    cout << "Generating Optimal Minimax Polynomials..." << endl;
    bootstrapper.prepare_mod_polynomial();
    cout << "Adding Bootstrapping Keys..." << endl;
    //bootstrapper.addBootKeys_3_other_slots(gal_keys, slot_vec);
	bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);
	keygen.create_galois_keys(gal_steps_vector, gal_keys);

    bootstrapper.slot_vec.push_back(logn);

	cout << "Generating Linear Transformation Coefficients..." << endl;
	bootstrapper.generate_LT_coefficient_3();

	Plaintext plain;
	Ciphertext cipher;

    vc mask;
    vector<vc> kv_cache;
    load_weight("mask",&mask,NULL,weights,encoder,encryptor,false);

    int i,j;
    int layers=1;

    vc embedded_input;
    vc current = embedded_input;
    vvec gamma,beta;

    char name[36];

    stringstream ss;
    string s;

    for(i=0;i<layers;i++){
        for(j=0;j<current.size();j++){
            load_weight(get_key(&ss,i,"transformer.h.",".ln_1.weight"),NULL,&gamma,weights,encoder,encryptor,false);
            load_weight(get_key(&ss,i,"transformer.h.",".ln_1.bias"),NULL,&beta,weights,encoder,encryptor,false);

            compute_layernorm(current[i],current[i],gamma[0],beta[0],768,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        }

        // Pack and bootstrap
        vc ln_pack_out;
        init_output(3,ln_pack_out,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        pack_tight(current,ln_pack_out,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        for(i=0;i<3;i++)
            bootstrapper.bootstrap_full_real_3(ln_pack_out[i],ln_pack_out[i]);

        // Pack back to rows for attn layer
        unpack_tight(ln_pack_out,current,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        
        vc qw,qb,kw,kb,vw,vb,w_out,b_out;
        load_weight(get_key(&ss,i,"transformer.h.",".attn.c_attn.weight_q"),&qw,NULL,weights,encoder,encryptor,true);
        load_weight(get_key(&ss,i,"transformer.h.",".attn.c_attn.bias_q"),&qb,NULL,weights,encoder,encryptor,true);
        load_weight(get_key(&ss,i,"transformer.h.",".attn.c_attn.weight_k"),&kw,NULL,weights,encoder,encryptor,true);
        load_weight(get_key(&ss,i,"transformer.h.",".attn.c_attn.bias_k"),&kb,NULL,weights,encoder,encryptor,true);
        load_weight(get_key(&ss,i,"transformer.h.",".attn.c_attn.weight_v"),&vw,NULL,weights,encoder,encryptor,true);
        load_weight(get_key(&ss,i,"transformer.h.",".attn.c_attn.bias_v"),&vb,NULL,weights,encoder,encryptor,true);
        load_weight(get_key(&ss,i,"transformer.h.",".attn.c_proj.weight"),&w_out,NULL,weights,encoder,encryptor,true);
        load_weight(get_key(&ss,i,"transformer.h.",".attn.c_proj.bias"),&b_out,NULL,weights,encoder,encryptor,true);


        vector<vc> kv_cache;
        attentionLayer(current,
                        qw,qb[0],kw,kb[0],vw,vb[0],w_out,b_out[0],mask[0],kv_cache,
                        current,128,768,i,
                        encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        
        // Pack and bootstrap
        vc attn_pack_out;
        init_output(3,attn_pack_out,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        pack_tight(current,attn_pack_out,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        for(i=0;i<3;i++)
            bootstrapper.bootstrap_full_real_3(ln_pack_out[i],ln_pack_out[i]);
        

        unpack_tight(attn_pack_out,current,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        // Feed-Forward Layer
        vc W1,b1,W2,b2;
        load_weight(get_key(&ss,i,"transformer.h.",".attn.c_fc.weight"),&W1,NULL,weights,encoder,encryptor,true);
        load_weight(get_key(&ss,i,"transformer.h.",".attn.c_fc.bias"),&b1,NULL,weights,encoder,encryptor,true);
        load_weight(get_key(&ss,i,"transformer.h.",".attn.c_proj.weight"),&W2,NULL,weights,encoder,encryptor,true);
        load_weight(get_key(&ss,i,"transformer.h.",".attn.c_proj.bias"),&b2,NULL,weights,encoder,encryptor,true);
        
        FeedForwardLayer(current,W1,b1,W2,b2,current,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);

        vc ff_pack_out;
        // Pack and Bootstrap
        pack_tight(current,ff_pack_out,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
        for(i=0;i<3;i++)
            bootstrapper.bootstrap_full_real_3(ff_pack_out[i],ff_pack_out[i]);
        unpack_tight(ff_pack_out,current,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    }

    // Final LayerNorm
    for(i = 0;i<current.size();i++){
        load_weight("transformer.h.ln_f.weight",NULL,&gamma,weights,encoder,encryptor,false);
        load_weight(name,NULL,&beta,weights,encoder,encryptor,false);
        compute_layernorm(current[i],current[i],gamma[0],beta[0],768,encoder,encryptor,decryptor,evaluator,gal_keys,relin_keys);
    }

    // Clear all memory used by protobufs
    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}


/*
# End-to-End GPT-2 Inference
# Assumes Pre-embedded input
# Outputs numpy version of output 
def gpt2_inference(embed_in,config,weights):

    assert embed_in.shape == (128,768)
    layer_in = pack.pack_from_row(embed_in)
    
    print("\nBeginning GPT2-CKKS inference","-"*50,"\n")
    mask = torch.tril(torch.ones((config.max_len, config.max_len)))
    mask2 = torch.triu(torch.ones((config.max_len, config.max_len)),diagonal=1)
    
    mask_packed = pack.pack_from_row(mask2)
    ex_mask_packed = pack.pack_from_row(mask)
    for idx in range(config.n_layers):

        print(f"LAYER IN: {layer_in[0][:10]}")
        # Layer Norm
        ln1_res = layers.layer_norm(layer_in,
                                    weights[f"transformer.h.{idx}.ln_1.weight"],
                                    weights[f"transformer.h.{idx}.ln_1.bias"],
                                    768,40298902)

        print(f"RES LN1 OUT: {ln1_res[0][:10]}")
        # Attn Layer
        attn_res = attn.attention_layer(ln1_res,
            weights[f"transformer.h.{idx}.attn.c_attn.weight_q"],
            weights[f"transformer.h.{idx}.attn.c_attn.bias_q"],
            weights[f"transformer.h.{idx}.attn.c_attn.weight_k"],
            weights[f"transformer.h.{idx}.attn.c_attn.bias_k"],
            weights[f"transformer.h.{idx}.attn.c_attn.weight_v"],
            weights[f"transformer.h.{idx}.attn.c_attn.bias_v"],
            weights[f"transformer.h.{idx}.attn.c_proj.weight"],
            weights[f"transformer.h.{idx}.attn.c_proj.bias"],
            mask_packed,ex_mask_packed
        )

        print(f"RES ATTN OUT: {attn_res[0][:10]}")

        print(f"LAYER IN: {layer_in[0][:10]}")
        # Residual Layer
        for i in range(len(ln1_res)):
            attn_res[i] += layer_in[i]
        
        print(f"RES RESID OUT: {attn_res[0][:10]}")
        
        # Second LayerNorm
        ln2_res = layers.layer_norm(attn_res,
                                    weights[f"transformer.h.{idx}.ln_2.weight"],
                                    weights[f"transformer.h.{idx}.ln_2.bias"],
                                    768,5.74156342e+08)

        print(f"RES LN 2 OUT: {ln2_res[0][:10]}")

        # MLP
        mlp_res = layers.mlp(ln2_res,
                             weights[f"transformer.h.{idx}.mlp.c_fc.weight"],
                             weights[f"transformer.h.{idx}.mlp.c_fc.bias"],
                             weights[f"transformer.h.{idx}.mlp.c_proj.weight"],
                             weights[f"transformer.h.{idx}.mlp.c_proj.bias"])
        print(f"RES MLP OUT: {mlp_res[0][:10]}")
        print(f"ATTN MLP OUT: {attn_res[0][:10]}")

        # Residual layer
        for i in range(len(mlp_res)):
            layer_in[i] = mlp_res[i] + attn_res[i]

        print(f"RES RESID 2 OUT: {layer_in[0][:10]}")

    # Final Layer norm
    out = layers.layer_norm(layer_in,
                                    weights["transformer.ln_f.weight"],
                                    weights["transformer.ln_f.bias"],
                                    768,1.1047115e+10)
    return out
*/