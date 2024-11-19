import matplotlib.pyplot as plt
import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp( - z)) 
plt.plot(np.arange(-5, 5, 0.1), sigmoid(np.arange(-5, 5, 0.1))) 
plt.title('Visualization of the Sigmoid Function') 
plt.show()
