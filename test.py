import numpy as np
from scipy.special import comb
a=np.array([[1,2],[3,4]])
b=np.array([3,1])

print(a-b)
print((a-b)**2)
print(a.sum(axis=1))

p=0.5
N=100
k=50
print(comb(N, k) * p ** k * (1 - p) ** (N - k))