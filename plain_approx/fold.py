import math
import numpy as np
import iterations as iters
import poly



def quickMax(vec,n):
    for i in range(math.floor(math.log(n,2))-1):
        rot_vec =  np.roll(vec,-1* 2**i)
        vec = poly.approx_max(vec,rot_vec)

    return vec

def quickSum(vec,n):
    for i in range(math.floor((math.log(n,2)))-1):
        rot_vec =  np.roll(vec,-1* 2**i)
        vec = vec + rot_vec
    return vec



if __name__ == "__main__":
    test1  = [.11111,.11112,.11113,.11114]*2
    test2 = [-0.1,2,0,-300,1,54,32,23.2]*2

    print("quickMax1",f"{test1}",quickMax(test1,4),max(test1))
    print("quckMax2",f"{test2}",quickMax(test2,4),max(test2))
    print("quickSum",f"{test1}",quickSum(test1,len(test1)),sum(test1)/2)
    print("quickSum",f"{test2}",quickSum(test2,len(test2)),sum(test2)/2)