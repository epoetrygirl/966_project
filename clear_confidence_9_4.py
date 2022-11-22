import numpy as np
import sys

seed = int(sys.argv[1])
class0 = int(sys.argv[2])
class1 = int(sys.argv[3])

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
clear_9 = [9, 12, 58, 99, 1058, 4009, 4030, 4047]
clear_4 = [19, 27, 49, 56, 295, 1080, 4042, 4046]

conf4 = AC.accuracy(x_test_all[clear_4], y_test_all[clear_4])[1]
conf9 = AC.accuracy(x_test_all[clear_9], y_test_all[clear_9])[1]

with open('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/clear_conf_scores_4.npy', 'wb') as f:
    np.save(f, conf4)

with open('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/clear_conf_scores_9_4.npy', 'wb') as g:
    np.save(g, conf9)

print("Confidence for 4:",conf4)
print("Confidence for 9:",conf9)
