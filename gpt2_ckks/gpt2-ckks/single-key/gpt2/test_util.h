#pragma once

#include "util.h"

#include<vector>

using namespace std;

enum TestType {
    ATTN_PROJ_ROW,
    QK_MATMUL,
    SV_MATMUL,
    SOFTMAX,
    GELU,
    LAYERNORM,
    BOOTSTRAP
};

typedef struct test_entry{
    string name;
    string description;
    TestType type;
} test_entry_t;

// Assume |v1| == |v2|
//double dot_prod(vector<double> &v1, vector<double> &v2);

// Assume A1 is m x n, A2 is n x k and A_out is m x k 
void matrix_mul(vector<vector<double>> &A1,vector<vector<double>> &A2,vector<vector<double>> &A_out);

// Assume A is m x n, A_t is n x m
void transpose(vector<vector<double>> &A,vector<vector<double>> &A_t);

// Apply e^x to all elements in A
void compute_exp_plain(vector<double> &A);

// Compute numberically stable softmax plain
void compute_softmax_plain(vector<double> &A,vector<double> &out);

// Randomly generate random numbers matching an input dimension
void generate_random(vector<vector<double>> &v);

// Adds descriptions of micorbenchmarks
void populate_tests(map<int,test_entry_t> &tests);

// Prints microbenchmark descriptions
void print_tests(map<int,test_entry_t> &tests);