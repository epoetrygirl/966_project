import numpy as np
from scipy.stats import median_test

baseline_recur = '0wwo2-raw_data_recur_orig_results_ba copy.txt'
increased_recur = '0-wwo-0raw_data_recur_in_results_ba copyfirst.txt'

baseline = np.zeros((10,10,20))
increased = np.zeros((10,10,20))

with open(baseline_recur) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
for line in lines:
    # print(int(line[]))
    class0 = int(line[26])
    class1 = int(line[29])
    seed = int(line[32])        
    baseline[class0][class1][seed] = float(line[-7:-2])

# print(baseline[0][1])

with open(baseline_recur) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
for line in lines:
    class0 = int(line[26])
    class1 = int(line[29])
    seed = int(line[32])        
    increased[class0][class1][seed] = float(line[-7:-2])

for i in range(10):
    for j in range(10):
        if i<j:
            p_values[i][j] = median_test(baseline[i][j], increased[i][j])[1]

# print(p_values)


import seaborn as sb
import matplotlib.pyplot as plt
ax = plt.axes()
mask =  np.tri(p_values.shape[0], k=0)
# meds = np.ma.array(meds, mask=mask)
heatmap = sb.heatmap(p_values, cmap="BuGn",vmin=0, vmax=0.05,annot=True,fmt='.2f',annot_kws={"size": 7},mask=mask)
ax.set_title("p-values from Mood's Median Test on Corresponding Medians" )
plt.show()




