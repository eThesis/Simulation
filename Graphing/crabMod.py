from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from close import *

data1 = np.random.normal(0.634, 0.01841195, 1000)
data2 = np.random.normal(0.657, 0.01256981, 1000)
i, j = int(2000 * 0.441 // 1), int(2000 * 0.559 // 1)
data3 = np.concatenate((
                            data1[:i],
                            data2[:j]
                        ))


#plt.hist(data1, bins=100, alpha=0.5, label="data1")
#plt.hist(data2, bins=100, alpha=0.5, label="data2")
plt.hist(data3, bins=100, alpha=0.5, label="data3", color='g')

plt.ylabel('Frequency')
plt.xlabel('Forehead to Body Length Ratio')
plt.title('Theoretical Mixture')

plt.xticks([], [])
plt.yticks([], [])
plt.box(False)

plt.show(block=False)
close('CrabModMix')
