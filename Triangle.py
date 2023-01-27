import math
import random
import numpy as np
import scipy as sp
import scipy.ndimage
from matplotlib import cm
import matplotlib.pyplot as plt

############################################################ deklaracja funkcji

def PrepareWeights(weights):
   return weights / np.sum(weights[:])

def Smoothening(tab, weights):
   return sp.ndimage.convolve(tab, weights, mode='constant')

def Distance(p1,p2):
   return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (landscape[p2] - landscape[p1])**2)

def MakeTriangle(p1,p2,p3, r, iter):
   iter -= 1
   
   tab = []
   tab.append(random.random() * math.pow(math.dist(p1, p2), r))
   tab.append(random.random() * math.pow(math.dist(p1, p3), r))
   tab.append(random.random() * math.pow(math.dist(p2, p3), r))

   for i in range(3):
      if random.randint(0, 1) == 0:
         tab[i] = -tab[i]
   
   p1_2 = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
   p1_3 = (int((p1[0]+p3[0])/2), int((p1[1]+p3[1])/2))
   p2_3 = (int((p2[0]+p3[0])/2), int((p2[1]+p3[1])/2))

   if landscape[p1_2] == 0:
      landscape[p1_2] = (landscape[p1] + landscape[p2])/2 + tab[0]
   if landscape[p1_3] == 0:
      landscape[p1_3] = (landscape[p1] + landscape[p3])/2 + tab[1]
   if landscape[p2_3] == 0:
      landscape[p2_3] = (landscape[p2] + landscape[p3])/2 + tab[2]

   if (iter == 0):
      return

   MakeTriangle(p1,p1_2,p1_3, r, iter)
   MakeTriangle(p2,p1_2,p2_3, r, iter)
   MakeTriangle(p3,p1_3,p2_3, r, iter)
   MakeTriangle(p1_2,p1_3,p2_3, r, iter)

def FillArray():
   for x in range(1, 2**N + 1):
      for y in range(x):
         landscape[x][y] = landscape[y][x]

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

f = plt.figure()
ax = plt.axes(projection="3d")
N = 9
R = 0.7

############################################################ Generacja krajobrazu

landscape = np.zeros(shape = (2**N + 1, 2**N + 1))

p1 = (0, 0)
p2 = (0, 2**N)
p3 = (2**N, 2**N)

landscape[p1] = random.random() * 2**N
landscape[p2] = random.random() * 2**N
landscape[p3] = random.random() * 2**N

MakeTriangle(p1,p2,p3, R, N)
FillArray()
PrepareWeights(weights)
landscape = Smoothening(landscape, weights)

############################################################ losowe podstawa

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

############################################################ kolorowanie oraz wyswietlanie

x_data = np.arange(0, 2**N + 1, 1)
y_data = np.arange(0, 2**N + 1, 1)

X, Y = np.meshgrid(x_data, y_data)

f.set_figwidth(20)
f.set_figheight(20)
X, Y = np.meshgrid(x_data, y_data)
coloring = cm.terrain(landscape/np.amax(landscape))
ax.plot_surface(X, Y, landscape, facecolors=coloring, cstride=1, rstride=1)
plt.show()
# plt.savefig('plot3.jpg')