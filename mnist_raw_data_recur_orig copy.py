import numpy as np
import sys
import os
from assembly_classifier import AssemblyClassifier

# seed = int(sys.argv[1])
class0 = int(sys.argv[1])
class1 = int(sys.argv[2])
# recur_in = int(sys.argv[4])

# recur_in = int(sys.argv[1])
# if recur_in == 0:
#     from assembly_classifier import AssemblyClassifier
# if recur_in == 1:

# raw pixels
x_train_all = np.load('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/binary-mnist/original_28x28/all_digits_all_pixels/x_train.npy')
y_train_all = np.load('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/binary-mnist/original_28x28/all_digits_all_pixels/y_train.npy')
x_test_all = np.load('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/binary-mnist/original_28x28/all_digits_all_pixels/x_test.npy')
y_test_all = np.load('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/binary-mnist/original_28x28/all_digits_all_pixels/y_test.npy')

# original data
# x_train_all = np.load('data/mnist_data/x_train.npy')
# x_test_all = np.load('data/mnist_data/x_test.npy')
# y_train_all = np.load('data/mnist_data/y_train.npy')
# y_test_all = np.load('data/mnist_data/y_test.npy') 

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
  
for i in range(10):
    np.random.seed(i)   
    x_train, y_train, x_test, y_test = extract_mnist_dataset(class0,class1)
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
    print(class0, class1, i,'Accuracy of the network on the test set: %d / %d = %.2f %%' % (correct, total, 100.*correct/total))
    
    # Write results to file
    writepath = '/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/raw_data_recur_orig_results.txt'
    mode = 'a' if os.path.exists(writepath) else 'w'
    with open(writepath, mode) as f:
        f.write('(class0, class1, seed) = (%d, %d, %d). Test set accuracy: %d / %d = %.2f %% \n' % (class0, class1, i, correct, total, 100.*correct/total))

# with open('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/conf_scores_7.npy', 'wb') as f:
#     np.save(f, AC.accuracy(x_test[y_test==0], y_test[y_test==0])[1])

# with open('/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/conf_scores_9_7.npy', 'wb') as g:
#     np.save(g, AC.accuracy(x_test[y_test==1], y_test[y_test==1])[1]) 
                
                
