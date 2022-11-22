import numpy as np
from assembly_classifier import AssemblyClassifier

np.random.seed(12)

x_train_all = np.load('data/mnist_data/x_train.npy')
x_test_all = np.load('data/mnist_data/x_test.npy')
y_train_all = np.load('data/mnist_data/y_train.npy')
y_test_all = np.load('data/mnist_data/y_test.npy')

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

x_train, y_train, x_test, y_test = extract_mnist_dataset(0,1)
d = x_train.shape[1]
AC = AssemblyClassifier(d, n_assembly=1000, n_cap=41, edge_prob=0.01, initial_projection=True)

correct = AC.accuracy(x_test, y_test)
total = len(y_test)
print("-----PRE-TRAINING-----")
print('Accuracy of the network on the test set: %d / %d = %.2f %%' % (correct, total, 100.*correct/total))

print("-----TRAINING-----")
n_train = 2000
AC.train(x_train[:n_train], y_train[:n_train])

correct = AC.accuracy(x_test, y_test)
total = len(y_test)
print("-----POST-TRAINING-----")
print('Accuracy of the network on the test set: %d / %d = %.2f %%' % (correct, total, 100.*correct/total))
