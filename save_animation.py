import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

a = np.random.random((45, 45))
im = plt.imshow(a, cmap='hot', interpolation='nearest', animated=True)

def updatefig(*args):
    im.set_array(np.random.random((45, 45)))
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show() 
