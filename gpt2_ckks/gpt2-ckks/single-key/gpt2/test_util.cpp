#include "test_util.h"
#include "math.h"
#include<algorithm>
#include <bits/stdc++.h>


void matrix_mul(vector<vector<double>> &A1,vector<vector<double>> &A2,vector<vector<double>> &A_out) {
    int m = A1.size(),n = A1[0].size(),k=A2[0].size();
    for(int i = 0;i<m;i++){
        for(int j = 0; j<k;j++){
            for(int c=0;c<n;c++){
                A_out[i][j] += A1[i][c] * A2[c][j];
            }
        }
    }
}

void transpose(vector<vector<double>> &A,vector<vector<double>> &A_t) {
    int i,j,m=A.size(),n=A[0].size();
    for(i=0;i<m;i++)
        for(j=0;j<n;j++)
            A_t[j][i]=A[i][j];
}

// Apply e^x to all elements in A
void compute_exp_plain(vector<double> &A){
    transform(A.begin(), A.end(), A.begin(), [](double num){return exp(num);});
}

// Compute numberically stable softmax
void compute_softmax_plain(vector<double> &A,vector<double> &out) {
    double max_ele,sum;

    max_ele = * max_element(A.begin(),A.end());

    transform(A.begin(),A.end(),out.begin(),[max_ele](double num){return num-max_ele;});

    sum = accumulate(out.begin(),out.end(),0);

    transform(out.begin(),out.end(),out.begin(),[sum](double num){return num/sum;});
}