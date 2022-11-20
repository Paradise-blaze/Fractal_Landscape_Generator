import random
import numpy as np

def FillArray():
   for i in range(N):
      Diamond()
      Square()

def Diamond():

def Square():


N = 4
landscape = np.zeros(shape = (2**N +1, 2**N + 1))

landscape[0,0] = random.random() * 100
landscape[0,2**N] = random.random() * 100
landscape[2**N,0] = random.random() * 100
landscape[2**N, 2**N] = random.random() * 100

FillArray()
print(landscape)