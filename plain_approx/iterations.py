import math
import numpy as np

# Third order taylor poly
def taylor_expand(x,a):
    coeffs = [(-0.5,-1.5),(-1.5*-0.5,-2.5),(-2.5*-1.5*-0.5,-3.5)]
    out = np.ones_like(x,dtype=complex) *(1/np.sqrt(a-1))
    start_minus_one = np.array(a-1,dtype=complex)
    for i,(coeff,power) in enumerate(coeffs):
        #print(i,out)
        out += coeff * np.power(start_minus_one,power) * np.power((x-a),i+1) * (1/math.factorial(i+1))
        #print(i,out)
    return out

def newton_iteration(a,start,iters):
    init_val = taylor_expand(a,start)
    #print(f"INIT VAL: {init_val}")
    x = init_val
    for _ in range(iters):
        x = x*(1.5-0.5*a*(x**2))
    return x

def goldschmidt_division(n,d,iters):
    for _ in range(iters):
        f = 2 - d
        n = n * f
        d = d * f
        #print(n.max(),d.max())
    return n

def exp(x,r):
    return np.power(1+(x/(math.pow(2,r))),math.pow(2,r))

print(exp(np.array([1,2,3]),6),math.e ** 2)

n_iters = 13
d_iters = 9

a = 20
n=1
d=0.0035


if __name__ == "__main__":
    #print("Newton",1/math.sqrt(20),newton_iteration(20,n_iters))
    #print("Newton",1/math.sqrt(100),newton_iteration(100,n_iters))
    args=np.array([3,5,7])
    print("Taylor",1/np.sqrt(args-1),taylor_expand(np.array([3,5,7]),9))

    args = np.array([1000,1200,800])
    print("Newton",1/np.sqrt(args-1),newton_iteration(np.array(args),1000,n_iters))

    print("Goldschmidt",n/d,goldschmidt_division(n,d,d_iters))
    print("Goldschmidt",n/0.40,goldschmidt_division(n,0.4,d_iters))
    print("Goldschmidt",n/0.67,goldschmidt_division(n,0.67,d_iters))

    r=10
    print("exp",math.exp(2),exp(2,r))
    print("exp",math.exp(-0.05),exp(-0.05,r))
    print("exp",math.exp(10),exp(10,r))
    print("exp",math.exp(7.39543214e-06),exp(np.array([7.39543214e-06,20.0]),r))

