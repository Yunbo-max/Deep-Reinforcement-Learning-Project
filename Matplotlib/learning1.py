import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-3,3,50)
y1 = 2*x+1
y2 = x**2


plt.figure()
plt.plot(x,y1)
plt.plot(x,y2,color='red',linewidth=1.0,linestyle='--')
plt.show()
plt.xlim((-5,-2))
plt.ylim((6,10))
plt.xlabel('number of monkey')
plt.ylabel('appearance')


