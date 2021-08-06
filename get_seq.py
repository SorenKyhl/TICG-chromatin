import numpy as np

def get_seq(n): 
    vec = np.zeros(n)
    vec[0] = 0
    print("%d" % vec[0])
    for i in range(1,n):
        if vec[i-1] == 1:
            vec[i] = np.random.choice([1,0], p=[0.95,0.05])
        else:
            vec[i] = np.random.choice([1,0], p=[0.05,0.95])
        print("%d" % vec[i])
            
    #return np.array(vec, dtype=int)


if __name__ ==  "__main__":
    get_seq(1024)
