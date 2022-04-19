import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from close import *

data = pd.read_csv('newCrab.csv', sep=',',header=None, index_col =0)
data1 = np.random.normal(0.634, 0.01841, 1000)
data2 = np.random.normal(0.657, 0.0126,1000)

data.plot(kind='bar')
#plt.hist(data1, bins=100, alpha=0.5, label="data1")
#plt.hist(data2, bins=100, alpha=0.5, label="data2")

plt.ylabel('Frequency')
plt.xlabel('Forehead to Body Length Ratio')
plt.title('Pearson Crab Data')

plt.xticks([], [])
plt.yticks([], [])
plt.box(False)
plt.gca().legend_.remove()




plt.show(block=False)
close('CrabRaw')