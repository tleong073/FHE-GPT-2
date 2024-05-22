import torch
import math
import numpy as np
def f(x):
    u = -0.00036515543567082031
    u = u * x + 1.8871903376027265e-119
    u = u * x + 0.0098482040825846138
    u = u * x + -9.1246190287405663e-119
    u = u * x + -0.095289016801412996
    u = u * x + 1.125251657756392e-118
    u = u * x + 0.50024220616613724
    u = u * x + 0.5
    return u * x + -0.0084029555638412776

def gelu_p(x):
    u = -0.010674138350676401
    u = u * x + -0.11491758706060971
    u = u * x + -0.4134048031372351
    return u * x + -0.49783059647700406

def gelu_q(x):
    u = 0.0012264627004247512
    u = u * x + 0.0020872891783959252
    u = u * x + -0.036434980917200932
    u = u * x + -0.0085812866991243648
    u = u * x + 0.36217359096393054
    u = u * x + 0.50622871052887408
    return u * x + 0.0072415619838619525

def gelu(x):
    s2,s1,s0 = 0.5*np.sign(x-3),0.5*np.sign(x+1.95),0.5*np.sign(x+4)

    b0,b1,b2,b3 = 0.5*s0,s0-s1,s1-s2,0.5*s2

    return (b0*0) + (b1*gelu_p(x)) + (b2*gelu_q(x)) + (b3*x)


def sign_f(x):
    u = -0.010674138350676401
    u = u * x + -0.11491758706060971
    u = u * x + -0.4134048031372351
    return u * x + -0.49783059647700406

def sign_q(x):
    u = 0.0012264627004247512
    u = u * x + 0.0020872891783959252
    u = u * x + -0.036434980917200932
    u = u * x + -0.0085812866991243648
    u = u * x + 0.36217359096393054
    u = u * x + 0.50622871052887408
    return u * x + 0.0072415619838619525

def evaluate_poly(x,coeffs):
    cur = np.ones_like(x)
    acc = np.zeros_like(x,dtype=float)
    for coeff in coeffs:
        acc += cur * coeff
        cur *= x
        #print(cur)
    return acc
         
def sign(x):
    sign_f = [35/128, 0,-180/128, 0, 378/128, 0, -420/128, 0 ,315/128, 0][::-1]
    sign_g = [46623/1024, 0, -113492/1024, 0, 97015/1024, 0, -34974/1024, 0, 5850/1024, 0][::-1]
    #print("sign g ",sign_g)
    post_g = evaluate_poly(evaluate_poly(x,sign_g),sign_g)
    #print("post g",post_g)
    return evaluate_poly(evaluate_poly(post_g,sign_f),sign_f)

def approx_max(a,b):
    c = a + b
    diff = a - b
    diff_normalized = diff/1000000
    signs = sign(diff_normalized)
    #print("MAX: ", np.abs(diff_normalized).max(),np.abs(diff_normalized).min())
    #assert (np.abs(diff_normalized) < 1).all()
    res = 0.5 * (c + ((diff) * signs))
    #print("RES: ", res[:5],signs[:5])
    return res



def gelu_bolt(x):
    a = 0.020848611754127593
    b = -0.18352506127082727
    c = 0.5410550166368381
    d = -0.03798164612714154
    e = 0.001620808531841547
    return a * (abs(x) ** 4) + b * (abs(x) ** 3) + c * (abs(x) ** 2) + d * (abs(x)) + e + 0.5* x

'''
gelu = torch.nn.GELU(approximate='tanh')
total_err = 0
maxi = 0.0
for i,v in enumerate([x/1000 for x in range(-1950,3000)]):
    with torch.no_grad():
        err = abs(gelu.forward(torch.tensor(v)) - f2(v))
        maxi=max(err,maxi)
        print(f"err: {err}")
        total_err += err
        print(f"total: {total_err}")
        print(total_err/i,"maximum err: ",maxi)
'''

if __name__ == "__main__":
    print(gelu_p(1))
    print(gelu_q(5))


    print(sign(np.array([-0.4000000,0.5000000000,0.01,-0.02])))
    a = np.array([.1,.2,.3,.4,.6])
    b = np.array([.3,-.2,.4,.5,.7])
    print(approx_max(a,b))

    gelu = torch.nn.GELU(approximate='tanh')
    total_err = 0
    maxi = 0.0
    for i,v in enumerate([x/1000 for x in range(-4000,-1950)]):
        with torch.no_grad():
            err = abs(gelu.forward(torch.tensor(v)) - gelu_p(v))
            maxi=max(err,maxi)
            print(f"err: {err}")
            total_err += err
            print(f"total: {total_err}")
            print(total_err/i,"maximum err: ",maxi)