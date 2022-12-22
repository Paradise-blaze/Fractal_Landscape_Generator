import math
import random
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


def Distance(p1,p2):
   return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (landscape[p2] - landscape[p1])**2)

def MakeTriangle(p1,p2,p3, r, iter):
   iter -= 1
   
   tab = []
   tab.append(random.random() * (Distance(p1, p2)**r))
   tab.append(random.random() * (Distance(p1, p3)**r))
   tab.append(random.random() * (Distance(p2, p3)**r))
   for i in range(3):
      if random.randint(0, 1) == 0:
         tab[i] = -tab[i]
   
   p1_2 = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
   p1_3 = (int((p1[0]+p3[0])/2), int((p1[1]+p3[1])/2))
   p2_3 = (int((p2[0]+p3[0])/2), int((p2[1]+p3[1])/2))

   landscape[p1_2] = (landscape[p1] + landscape[p2]/2) + tab[0]
   landscape[p1_3] = (landscape[p1] + landscape[p3]/2) + tab[1]
   landscape[p2_3] = (landscape[p2] + landscape[p3]/2) + tab[2]

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
         
f = plt.figure()
ax = plt.axes(projection="3d")
N = 4
R = 0.6
landscape = np.zeros(shape = (2**N + 1, 2**N + 1))

p1 = (0, 0)
p2 = (0, 2**N)
p3 = (2**N, 2**N)

landscape[p1] = random.random() * 100
landscape[p2] = random.random() * 100
landscape[p3] = random.random() * 100

MakeTriangle(p1,p2,p3, R, N)
FillArray()
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

X, Y = np.meshgrid(x_data, y_data)

f.set_figwidth(20)
f.set_figheight(20)
X, Y = np.meshgrid(x_data, y_data)
coloring = cm.terrain(landscape/np.amax(landscape))
ax.plot_surface(X, Y, landscape, facecolors=coloring, cstride=1, rstride=1)
#plt.show()
plt.savefig('plot2.pdf')