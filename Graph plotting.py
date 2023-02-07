import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from math import exp





#2-D linear graph plotting

x=np.linspace(-2,10)
y1=2.4+(0*x)
y2=3*x+5
y3=-2.5*x+10
y4=x+11

plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize = (10,10))
plt.plot(x,y1,label="1st")
plt.plot(x,y2,label="2nd")
plt.plot(x,y3,label="3rd")
plt.plot(x,y4,label="4th")
plt.legend()
plt.show()



#2-D higher degree polynomial
x=np.linspace(-1,2)
y1=(6*(x**2))-(5*x)+2
y2=(x**3)-(2*(x**2))-1
y3=(4*(x**3))+(x**2)-3*x+5

fig = plt.figure(figsize = (10,10))
plt.plot(x,y1,label="1st")
plt.plot(x,y2,label="2nd")
plt.plot(x,y3,label="3rd")
plt.legend()
plt.show()


#3-D or 3 variables curve plotting
fig = plt.figure(figsize=(10, 10))
p = plt.axes(projection='3d')
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)

z1 = x + y + x * y + 3
z2 = 3.5 * x * y + 2 * (x ** 2) + 9 * y
z3 = ((x ** 2) * y) + (3 * x * y) - (7 * (y ** 2)) + 2.3
z4 = (5 * x * y) + 2
z5 = []

for x1, y1 in zip(x, y):
    z5.append(((5 * x1 * y1) / (exp((x1 ** 2) + (y1 ** 2)))) + 2)

p.plot3D(x, y, z1, 'blue')
p.plot3D(x, y, z2, 'green')
p.plot3D(x, y, z3, 'red')
p.plot3D(x, y, z4, 'black')
p.plot3D(x, y, z5, 'yellow')
plt.show()
