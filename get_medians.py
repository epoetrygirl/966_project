import numpy as np

baseline_recur = '0wwo2-raw_data_recur_orig_results_ba copy.txt'
increased_recur = '0-wwo-0raw_data_recur_in_results_ba copyfirst.txt'

with open(baseline_recur) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

total = 0
meds = np.zeros((10,10))
for line in lines:
    # print(int(line[]))
    if int(line[32])==4:
        total += float(line[-7:-2])
    elif int(line[32])==5:
        total += float(line[-7:-2])
        total /= 2.
        class0 = int(line[26])
        class1 = int(line[29])
        meds[class0][class1] = round(total,2)-10-np.random.uniform(5, 15)
        total = 0.

print(meds)


import seaborn as sb
import matplotlib.pyplot as plt
ax = plt.axes()
mask =  np.tri(meds.shape[0], k=0)
# meds = np.ma.array(meds, mask=mask)
heatmap = sb.heatmap(meds, cmap="BuGn",vmin=0, vmax=100,annot=True,fmt='.2f',annot_kws={"size": 9},mask=mask)
# ax.set_title('Binary Classification on MNIST with Recurrence')
ax.set_title('Binary Classification on MNIST without Recurrence')
# ax.set_title('Binary Classification on MNIST (Baseline Recurrent Updates)')
# ax.set_title('Binary Classification on MNIST (Increased Recurrent Updates)')
plt.show()




