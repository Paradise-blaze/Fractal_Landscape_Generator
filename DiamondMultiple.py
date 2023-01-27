import random
import numpy as np
import scipy as sp
import scipy.ndimage
from matplotlib import cm
import matplotlib.pyplot as plt

############################################################ potrzebne zmienne

f = plt.figure()
ax = plt.axes(projection="3d")
N = 8
BIGN = N+2
R = 64

landscapes = []
mediumLandscapes = []

wdth = 2 ** N

############################################################ deklaracja funkcji

def PrepareWeights(n):
   if n == 0:
      weights = np.array([[0, 1, 0],
                    [1, 2, 1],
                    [0, 1, 0]],
                   dtype=float)
   elif n == 1:
      weights = np.array([[0, 0, 1, 0, 0],
                    [0, 2, 4, 2, 0],
                    [1, 4, 8, 4, 1],
                    [0, 2, 4, 2, 0],
                    [0, 0, 1, 0, 0]],
                   dtype=float)
   elif n == 2:
      weights = np.array([[0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 2, 4, 2, 0, 0],
                    [0, 2, 4, 8, 4, 2, 0],
                    [1, 4, 8, 16, 8, 4, 1],
                    [0, 2, 4, 8, 4, 2, 0],
                    [0, 0, 2, 4, 2, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0]],
                   dtype=float)

   weights = weights / np.sum(weights[:])

   return weights

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

############################################################ Generacja i "zlepianie" krajobrazów

bigLandscape = np.zeros(shape=(2**BIGN+1, 2**BIGN+1))
landscape = np.zeros(shape=(2**N+1, 2**N+1))

subN = 2**(BIGN-N)
howMany = int(subN * subN)
print(howMany)
for i in range(howMany):
   landscape = np.zeros(shape=(2**N+1, 2**N+1))
   R = 64
   wdth = 2 ** N

   landscape[0, 0] = random.random() * 128
   landscape[0, 2**N] = random.random() * 128
   landscape[2**N, 0] = random.random() * 128
   landscape[2**N, 2**N] = random.random() * 128

   FillArray()
   landscapes.append(landscape.copy())

for i in range((subN)):
   mediumLandscapes.append(np.concatenate(landscapes[(subN)*i:(subN)*i+(subN)]))
bigLandscape = np.concatenate(mediumLandscapes,axis=1)

weights = PrepareWeights(1)
bigLandscape = Smoothening(bigLandscape, weights)

############################################################ losowe podstawa


x_data = np.arange(0, 2**BIGN + (subN), 1)
y_data = np.arange(0, 2**BIGN + (subN), 1)

X, Y = np.meshgrid(x_data, y_data)

base = np.zeros(shape=(2**BIGN+(subN), 2**BIGN+(subN)))

base[0, 0] = random.random() * 20
base[0, 2**BIGN] = random.random() * 20
base[2**BIGN, 0] = random.random() * 20
base[2**BIGN, 2**BIGN] = random.random() * 20

for x in range(0,(2**BIGN+(subN))):
   if x not in [0,2**BIGN]:
      base[x][0] = base[0][0]*(2**BIGN-x)/2**BIGN + base[2**BIGN][0]*x/2**BIGN
      base[x][2**BIGN] = base[0][2**BIGN]*(2**BIGN-x)/2**BIGN + base[2**BIGN][2**BIGN]*x/2**BIGN
   for y in range(0,(2**BIGN+(subN))):
      if y not in [0, 2 ** BIGN]:
         base[x][y] = base[x][0]*(2**BIGN-y)/2**BIGN + base[x][2**BIGN]*y/2**BIGN

for x in range(0,(2**BIGN+(subN))):
   for y in range(0, (2**BIGN+(subN))):
      bigLandscape[x][y] += base[x][y]

############################################################ kolorowanie oraz wyswietlanie

x_data = np.arange(0, 2**BIGN + (subN), 1)
y_data = np.arange(0, 2**BIGN + (subN), 1)
f.set_figwidth(10)
f.set_figheight(10)
X, Y = np.meshgrid(x_data, y_data)
coloring = cm.terrain(bigLandscape/np.amax(bigLandscape))
ax.plot_surface(X, Y, bigLandscape, facecolors=coloring, cstride=1, rstride=1)
# plt.show()
plt.savefig('plotBIGD11R128_1.jpg')