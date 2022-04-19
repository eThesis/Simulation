import numpy as np 
import matplotlib.pyplot as plt

sample_N, cluster_N, mixing_Prop = 1000, 3, [.1, .3, .6]

mean_1 = np.array([-10, -10])
cov_1 = np.array([
                    [5, 0],
                    [0, 5]
                ])

mean_2 = np.array([-5, 15])
cov_2 = np.array([
                    [12, 0],
                    [0, 12]
                ])

mean_3 = np.array([10, 0])
cov_3 = np.array([
                    [13, 3],
                    [3, 3]
                ])

title = 'simData'
plt.title(str(sample_N) + ' samples in ' + str(cluster_N) + ' clusters')
plt.xlabel('x axis')
plt.ylabel('y axis')

def plot(sample):
    for i in range(sample_N):
        if sample[i] == 0:
            X, Y = np.random.multivariate_normal(mean_1, cov_1).T
            plt.plot([X], [Y], marker='.')
        if sample[i] == 1:
            X, Y = np.random.multivariate_normal(mean_2, cov_2).T
            plt.plot([X], [Y], marker='.')
        else:
            X, Y = np.random.multivariate_normal(mean_3, cov_3).T
            plt.plot([X], [Y], marker='.')