import numpy as np
import sys

seed = int(sys.argv[1])
class0 = 4
class1 = 9

np.random.seed(seed) 

from assembly_classifier import AssemblyClassifier

# raw pixels
x_train_all = np.load('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/binary-mnist/original_28x28/all_digits_all_pixels/x_train.npy')
y_train_all = np.load('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/binary-mnist/original_28x28/all_digits_all_pixels/y_train.npy')
x_test_all = np.load('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/binary-mnist/original_28x28/all_digits_all_pixels/x_test.npy')
y_test_all = np.load('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/binary-mnist/original_28x28/all_digits_all_pixels/y_test.npy')

# original data
# x_train_all = np.load('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/data/mnist_data/x_train.npy')
# x_test_all = np.load('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/data/mnist_data/x_test.npy')
# y_train_all = np.load('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/data/mnist_data/y_train.npy')
# y_test_all = np.load('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/data/mnist_data/y_test.npy') 


def extract_mnist_dataset(zero_class, one_class):
    zeroidx = y_train_all == zero_class
    oneidx = y_train_all == one_class
    filteridx = zeroidx+oneidx
    x_train = x_train_all[filteridx]
    y_train = y_train_all[filteridx]

    zeroidx = y_test_all == zero_class
    oneidx = y_test_all == one_class
    filteridx = zeroidx+oneidx
    x_test = x_test_all[filteridx]
    y_test = y_test_all[filteridx]

    y_train[y_train==zero_class] = 0
    y_train[y_train==one_class] = 1
    y_test[y_test==zero_class] = 0
    y_test[y_test==one_class] = 1

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = extract_mnist_dataset(class0,class1)
d = x_train.shape[1]
AC = AssemblyClassifier(d, n_assembly=1000, n_cap=41, edge_prob=0.01, initial_projection=True)

correct, temp = AC.accuracy(x_test, y_test) 
total = len(y_test)
print("-----PRE-TRAINING-----")
print('Accuracy of the network on the test set: %d / %d = %.2f %%' % (correct, total, 100.*correct/total))


print("-----TRAINING-----")
n_train = 2000
AC.train(x_train[:n_train], y_train[:n_train])


print("-----POST-TRAINING-----")

conf_scores_9 = np.load('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/conf_scores_9_4.npy')
conf_scores_4 = np.load('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/conf_scores_4.npy')

# FIX
min4 = []
min9 = conf_scores_9.argsort()[:100][[8, 13, 17, 19, 20, 65, 67, 97]]
min9[min9>982] = 0

images_to_show = []
image_labels = []
for img in x_test[y_test==0][min9]:
  images_to_show.append(img)
for img in x_test[y_test==1][min4]:
  images_to_show.append(img)
  
for lbl in y_test[y_test==0][min9]:
  image_labels.append(lbl)
for lbl in y_test[y_test==1][min4]:
  image_labels.append(lbl)
 
confs_4_then_9 = AC.accuracy(images_to_show, image_labels)[1]

with open('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/ambig_9_4_conf_scores.npy', 'wb') as f:
    np.save(f, confs_4_then_9)

print("Confidence for Ambiguous 9/4 Images:",confs_4_then_9)