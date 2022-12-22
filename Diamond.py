import random
import numpy as np
import scipy as sp
import scipy.ndimage
from matplotlib import cm
import matplotlib.pyplot as plt

f = plt.figure()
ax = plt.axes(projection="3d")
N = 5
R = 64

wdth = 2 ** N

def PrepareWeights(weights):
   return weights / np.sum(weights[:])

def Smoothening(tab, weights):
   return sp.ndimage.convolve(tab, weights, mode='constant')

def FillArray():
   global wdth, R
   while wdth > 1:
      Diamond(wdth)
      Square(wdth)
      wdth //= 2
      R //= 2
      if R < 1:
         R = 1
def Diamond(wdth):
   for x in range(0, 2**N-1, wdth):
      for y in range(0, 2**N-1, wdth):
         middle = (landscape[x][y] + landscape[x+wdth][y] + landscape[x][y+wdth] + landscape[x+wdth][y+wdth]) / 4.0
         middle += random.random() * 2 * R - R
         landscape[x+wdth//2][y+wdth//2] = middle



def Square(wdth):
   for x in range(0, 2**N-1, wdth//2):
      for y in range((x+wdth//2) % wdth, 2**N-1, wdth):
         middle = (landscape[(x-wdth//2 + 2**N-1) % (2**N-1)][y] +
               landscape[(x+wdth//2) % (2**N-1)][y] +
               landscape[x][(y+wdth//2) % (2**N-1)] +
               landscape[x][(y-wdth//2 + 2**N-1) % (2**N-1)]) / 4.0

         middle += random.random() * 2 * R - R

         landscape[x][y] = middle

         if x == 0:
            landscape[2**N-1][y] = middle
         if y == 0:
            landscape[x][2**N-1] = middle


weights = np.array([[0, 0, 1, 0, 0],
                    [0, 2, 4, 2, 0],
                    [1, 4, 8, 4, 1],
                    [0, 2, 4, 2, 0],
                    [0, 0, 1, 0, 0]],
                    dtype=float)
# weights = np.array([[0, 1, 0],
#                     [1, 2, 1],
#                     [0, 1, 0]],
#                     dtype=float)

landscape = np.zeros(shape=(2**N+1, 2**N+1))

landscape[0, 0] = random.random() * 100
landscape[0, 2**N] = random.random() * 100
landscape[2**N, 0] = random.random() * 100
landscape[2**N, 2**N] = random.random() * 100

FillArray()
PrepareWeights(weights)
landscape = Smoothening(landscape, weights)


x_data = np.arange(0, 2**N + 1, 1)
y_data = np.arange(0, 2**N + 1, 1)

X, Y = np.meshgrid(x_data, y_data)

base = np.zeros(shape=(2**N+1, 2**N+1))

base[0, 0] = random.random() * 20
base[0, 2**N] = random.random() * 20
base[2**N, 0] = random.random() * 20
base[2**N, 2**N] = random.random() * 20

for x in range(0,(2**N+1)):
   if x not in [0,2**N]:
      base[x][0] = base[0][0]*(2**N-x)/2**N + base[2**N][0]*x/2**N
      base[x][2**N] = base[0][2**N]*(2**N-x)/2**N + base[2**N][2**N]*x/2**N
   for y in range(0,(2**N+1)):
      if y not in [0, 2 ** N]:
         base[x][y] = base[x][0]*(2**N-y)/2**N + base[x][2**N]*y/2**N

for x in range(0,(2**N+1)):
   for y in range(0, (2**N+1)):
      landscape[x][y] += base[x][y]


x_data = np.arange(0, 2**N + 1, 1)
y_data = np.arange(0, 2**N + 1, 1)
f.set_figwidth(20)
f.set_figheight(20)
X, Y = np.meshgrid(x_data, y_data)
coloring = cm.terrain(landscape/np.amax(landscape))
ax.plot_surface(X, Y, landscape, facecolors=coloring, cstride=1, rstride=1)
plt.show()
# plt.savefig('plot1s.jpg')