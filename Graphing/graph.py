import matplotlib.pyplot as plt
from numpy import block
from plot import *
from close import close

dataFile, points = 'new.txt', 50
gName, xName, yName = 'Gene Expression', 'Measurement', 'Concentration'

plt.title(gName)
plt.xlabel(xName)
plt.ylabel(yName)

## Start 

plot(dataFile, points)

## End

plt.show(block=False)
close(gName)







   

