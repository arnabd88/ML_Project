import numpy as np
from numpy import linalg 
#import scipy as sp
#import scipy.sparse as ssparse
#import scipy.sparse.linalg as slinalg

def main():
   # L = [{(1,2)}, {(2,3), (4,2), (5,1)}, {(0,3),(3,6),(6,2), (2,5)}, {(4,2), (1,3)}]
   # p = max([len(row) for row in L])
   # print p

    #A = ssparse.csr_matrix([[1, 0, 2], [0, 0, 3], [0, 1, 4], [3,1,0]])
    #B = ssparse.csr_matrix([[0, 0, 5], [1, 0, 7], [5, 1, 5]])

    #C = A.multiply(B)
    #print C

    A = [[1, 0, 2], [0, 0, 3], [0, 1, 4], [3,1,0]]
    B = [[0, 0, 5], [1, 0, 7], [5, 1, 5]]
    C = np.dot(A, B)

    print C

    D = linalg.pinv(C, 1e-15)

    print D

    C_T = C.transpose()
    M = np.dot(C_T, C)

    inv_M = linalg.inv(M)
    C_Dagger = np.dot(inv_M, C_T)

    print C_Dagger
    

if __name__ == "__main__":
    main()
