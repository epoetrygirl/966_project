import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os

# Creating two classes of stimuli of size k, class1 is supported
# on the first half of the neurons in the stimulus while class2
# is supported on the second half. We will "train" an area on stimuli
# from both classes and then project some unseen samples to create new
# assemblies.


def create_binary_dataset(stim_size, support, N):
    X1 = np.zeros((N, stim_size))
    X2 = np.zeros_like(X1)

    for i in range(N):
        idx1 = np.random.permutation(stim_size//2)[:support]
        idx2 = np.random.permutation(stim_size//2)[:support] + stim_size//2
        X1[i,idx1] = 1.
        X2[i,idx2] = 1.

    return np.vstack((X1, X2)), np.concatenate([np.zeros(N), np.ones(N)])

class Area:
    def __init__(self, n, k, p, beta=0.05):
        self._n = n
        self._k = k
        self._p = p
        self._beta = beta

    def project(self, stimulus, T, update=True, refresh=True):
        if hasattr(self, 'W_yx') is False or refresh:
            self.W_yx = np.random.binomial(1,self._p,size=(self._n,self._n)).astype("float64")
        if hasattr(self, 'W_yy') is False or refresh:
            self.W_yy = np.random.binomial(1,self._p,size=(self._n,self._n)).astype("float64")

        y_tm1 = np.zeros(self._n)

        for t in range(T):
            y_t = self.W_yx.dot(stimulus) + self.W_yy.dot(y_tm1)
            if len(np.where(y_t !=0)[0]) > self._k:
                indices = np.argsort(y_t)
                y_t[indices[:-self._k]]=0

            y_t[np.where(y_t != 0.)[0]] = 1.0
            conf = np.abs(sum(y_t[:self._n//2]) - sum(y_t[self._n//2:])) / self._k
            # y_tm1 = np.copy(y_t)

            if update:
                # plasticity modifications
                for i in np.where(y_t!=0)[0]:
                    for j in np.where(stimulus!=0)[0]:
                        self.W_yx[i,j] *= 1.+self._beta

                for i in np.where(y_t!=0)[0]:
                    for j in np.where(y_tm1!=0)[0]:
                        self.W_yy[i,j] *= 1.+self._beta

            y_tm1 = np.copy(y_t)
        return y_t, conf

    def project_learning_classes(self, stimulus_set, T, refresh=True):
        if refresh:
            self.W_yx = np.random.binomial(1,self._p,size=(self._n,self._n)).astype("float64")
            self.W_yy = np.random.binomial(1,self._p,size=(self._n,self._n)).astype("float64")

        num_samples = stimulus_set.shape[0]//2
        for c in [0,1]:
            sample_idxs = np.random.choice(num_samples, size=(T)) + c*num_samples
            y_tm1 = np.zeros(self._n)

            for t in range(T):
                y_t = self.W_yx.dot(stimulus_set[sample_idxs[t],:]) + self.W_yy.dot(y_tm1)
                if len(np.where(y_t !=0)[0]) > self._k:
                    indices = np.argsort(y_t)
                    y_t[indices[:-self._k]]=0

                y_t[np.where(y_t != 0.)[0]] = 1.0
                # y_tm1 = np.copy(y_t)

                # plasticity modifications
                for i in np.where(y_t!=0)[0]:
                    for j in np.where(stimulus_set[sample_idxs[t],:]!=0)[0]:
                        self.W_yx[i,j] *= 1.+self._beta

                for i in np.where(y_t!=0)[0]:
                    for j in np.where(y_tm1!=0)[0]:
                        self.W_yy[i,j] *= 1.+self._beta

                y_tm1 = np.copy(y_t)

    def reciprocal_project(self, stimulus, T, update=True, refresh=True):
        if hasattr(self, 'W_yx') is False or refresh:
            self.W_yx = np.random.binomial(1,self._p,size=(self._n,self._n)).astype("float64")
        if hasattr(self, 'W_yy') is False or refresh:
            self.W_yy = np.random.binomial(1,self._p,size=(self._n,self._n)).astype("float64")
        if hasattr(self, 'W_yx') is False or refresh:
            self.W_yz = np.random.binomial(1,self._p,size=(self._n,self._n)).astype("float64")
        if hasattr(self, 'W_yx') is False or refresh:
            self.W_zy = np.random.binomial(1,self._p,size=(self._n,self._n)).astype("float64")
        if hasattr(self, 'W_yx') is False or refresh:
            self.W_zz = np.random.binomial(1,self._p,size=(self._n,self._n)).astype("float64")

        y_tm1 = np.zeros(self._n)
        z_tm1 = np.zeros(self._n)

        for t in range(T):
            y_t = self.W_yx.dot(stimulus) + self.W_yy.dot(y_tm1) + self.W_yz.dot(z_tm1)
            if len(np.where(y_t !=0)[0]) > self._k:
                y_indices = np.argsort(y_t)
                y_t[y_indices[:-self._k]]=0

            y_t[np.where(y_t != 0.)[0]] = 1.0
            # y_tm1 = np.copy(y_t)

            if update:
                # plasticity modifications
                for i in np.where(y_t!=0)[0]:
                    for j in np.where(stimulus!=0)[0]:
                        self.W_yx[i,j] *= 1.+self._beta

                for i in np.where(y_t!=0)[0]:
                    for j in np.where(y_tm1!=0)[0]:
                        self.W_yy[i,j] *= 1.+self._beta

                for i in np.where(y_t!=0)[0]:
                    for j in np.where(z_tm1!=0)[0]:
                        self.W_yz[i,j] *= 1.+self._beta

            z_t = self.W_zy.dot(y_tm1) + self.W_zz.dot(z_tm1)
            if len(np.where(z_t !=0)[0]) > self._k:
                z_indices = np.argsort(z_t)
                z_t[z_indices[:-self._k]]=0

            z_t[np.where(z_t != 0.)[0]] = 1.0

            if update:
                # plasticity modifications
                for i in np.where(z_t!=0)[0]:
                    for j in np.where(y_tm1!=0)[0]:
                        self.W_zy[i,j] *= 1.+self._beta

                for i in np.where(z_t!=0)[0]:
                    for j in np.where(z_tm1!=0)[0]:
                        self.W_zz[i,j] *= 1.+self._beta

            y_tm1 = np.copy(y_t)
            z_tm1 = np.copy(z_t)
        return y_t, z_t

    def reciprocal_project_learning_classes(self, stimulus_set, T, refresh=True):
        if refresh:
            self.W_yx = np.random.binomial(1,self._p,size=(self._n,self._n)).astype("float64")
            self.W_yy = np.random.binomial(1,self._p,size=(self._n,self._n)).astype("float64")
            self.W_yz = np.random.binomial(1,self._p,size=(self._n,self._n)).astype("float64")
            self.W_zy = np.random.binomial(1,self._p,size=(self._n,self._n)).astype("float64")
            self.W_zz = np.random.binomial(1,self._p,size=(self._n,self._n)).astype("float64")

        num_samples = stimulus_set.shape[0]//2
        for c in [0,1]:
            sample_idxs = np.random.choice(num_samples, size=(T)) + c*num_samples
            y_tm1 = np.zeros(self._n)
            z_tm1 = np.zeros(self._n)

            for t in range(T):
                y_t = self.W_yx.dot(stimulus_set[sample_idxs[t],:]) + self.W_yy.dot(y_tm1) + self.W_yz.dot(z_tm1)
                if len(np.where(y_t !=0)[0]) > self._k:
                    y_indices = np.argsort(y_t)
                    y_t[y_indices[:-self._k]]=0

                y_t[np.where(y_t != 0.)[0]] = 1.0
                # y_tm1 = np.copy(y_t)

                # plasticity modifications
                for i in np.where(y_t!=0)[0]:
                    for j in np.where(stimulus_set[sample_idxs[t],:]!=0)[0]:
                        self.W_yx[i,j] *= 1.+self._beta

                for i in np.where(y_t!=0)[0]:
                    for j in np.where(y_tm1!=0)[0]:
                        self.W_yy[i,j] *= 1.+self._beta

                for i in np.where(y_t!=0)[0]:
                    for j in np.where(z_tm1!=0)[0]:
                        self.W_yz[i,j] *= 1.+self._beta

                z_t = self.W_zy.dot(y_tm1) + self.W_zz.dot(z_tm1)
                if len(np.where(z_t !=0)[0]) > self._k:
                    z_indices = np.argsort(z_t)
                    z_t[z_indices[:-self._k]]=0

                z_t[np.where(z_t != 0.)[0]] = 1.0
                # plasticity modifications
                for i in np.where(z_t!=0)[0]:
                    for j in np.where(y_tm1!=0)[0]:
                        self.W_zy[i,j] *= 1.+self._beta

                for i in np.where(z_t!=0)[0]:
                    for j in np.where(z_tm1!=0)[0]:
                        self.W_zz[i,j] *= 1.+self._beta

                y_tm1 = np.copy(y_t)
                z_tm1 = np.copy(z_t)

if __name__ == "__main__":
    n, k, p, N, B = 1000, 31, 0.1, 100, 0.1
    X_tr, labels_tr = create_binary_dataset(n,k,N)
    X_te, labels_te = create_binary_dataset(n,k,N)

    clf = LogisticRegression()
    clf.fit(X_tr, labels_tr)
    print("Test score for original representations: %.3f"%(clf.score(X_te,labels_te)))

    reps = 20
    test_acc_random = np.zeros(reps)
    train_acc_random = np.zeros(reps)
    test_acc_project = np.zeros(reps)
    train_acc_project = np.zeros(reps)
    test_acc_double_project = np.zeros(reps)
    train_acc_double_project = np.zeros(reps)

    for r in range(reps):
        area1 = Area(n,k,p, beta=B)
        P_X_tr = np.vstack([area1.project(X_tr[i,:],20)[0][np.newaxis,:] for i in range(2*N)])
        P_X_te = np.vstack([area1.project(X_te[i,:],20)[0][np.newaxis,:] for i in range(2*N)])
        # P_X_tr_conf = np.vstack([area1.project(X_tr[i,:],20)[1] for i in range(2*N)])
        P_X_te_conf = np.array([area1.project(X_te[i,:],20)[1] for i in range(2*N)])


        clf = LogisticRegression()
        clf.fit(P_X_tr, labels_tr)
        train_acc_random[r] = clf.score(P_X_tr,labels_tr)
        test_acc_random[r] = clf.score(P_X_te,labels_te)

        area2 = Area(n,k,p, beta=B)
        area2.project_learning_classes(X_tr, 20)
        Q_X_tr = np.vstack([area2.project(X_tr[i,:],10, update=False, refresh=False)[0][np.newaxis,:] for i in range(2*N)])
        Q_X_te = np.vstack([area2.project(X_te[i,:],10, update=False, refresh=False)[0][np.newaxis,:] for i in range(2*N)])
        # Q_X_tr_conf = np.vstack([area2.project(X_tr[i,:],10, update=False, refresh=False)[1] for i in range(2*N)])
        Q_X_te_conf = np.array([area2.project(X_te[i,:],10, update=False, refresh=False)[1] for i in range(2*N)])

        clf = LogisticRegression()
        clf.fit(Q_X_tr, labels_tr)
        train_acc_project[r] = clf.score(Q_X_tr,labels_tr)
        test_acc_project[r] = clf.score(Q_X_te,labels_te)

        area6 = Area(n,k,p, beta=B)
        area6.project_learning_classes(Q_X_tr, 40)
        QQ_X_tr = np.vstack([area6.project(Q_X_tr[i,:],20, update=False, refresh=False)[0][np.newaxis,:] for i in range(2*N)])
        QQ_X_te = np.vstack([area6.project(Q_X_te[i,:],20, update=False, refresh=False)[0][np.newaxis,:] for i in range(2*N)])
        # QQ_X_tr_conf = np.vstack([area6.project(Q_X_tr[i,:],20, update=False, refresh=False)[np.newaxis,:] for i in range(2*N)])
        QQ_X_te_conf = np.array([area6.project(Q_X_te[i,:],20, update=False, refresh=False)[1] for i in range(2*N)])

        clf = LogisticRegression()
        clf.fit(QQ_X_tr, labels_tr)
        train_acc_double_project[r] = clf.score(QQ_X_tr,labels_tr)
        test_acc_double_project[r] = clf.score(QQ_X_te,labels_te)

    print("Train score for project assemblies created randomly: %.3f (%.3f)"%(np.mean(train_acc_random), np.std(train_acc_random)))
    print("Train score for project assemblies created using learning algorithm: %.3f (%.3f)"%(np.mean(train_acc_project), np.std(train_acc_project)))
    print("Train score for project assemblies created using double projection learning algorithm: %.3f (%.3f)"%(np.mean(train_acc_double_project), np.std(train_acc_double_project)))

    print("Test score for project assemblies created randomly: %.3f (%.3f)"%(np.mean(test_acc_random), np.std(test_acc_random)))
    print("Test score for project assemblies created using learning algorithm: %.3f (%.3f)"%(np.mean(test_acc_project), np.std(test_acc_project)))
    print("Test score for project assemblies created using double projection learning algorithm: %.3f (%.3f)"%(np.mean(test_acc_double_project), np.std(test_acc_double_project)))

    print("Test conf for project assemblies created randomly:", np.mean(P_X_te_conf),np.std(P_X_te_conf))
    print("Test conf for project assemblies created using learning algorithm:", np.mean(Q_X_te_conf),np.std(Q_X_te_conf))
    print("Test conf for project assemblies created using double projection learning algorithm:",np.mean(QQ_X_te_conf),np.std(QQ_X_te_conf))

    print("Test conf for project assemblies created randomly:", P_X_te_conf)
    print("Test conf for project assemblies created using learning algorithm:", Q_X_te_conf)
    print("Test conf for project assemblies created using double projection learning algorithm:",QQ_X_te_conf)

    writepath = '/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/results_notshuffled_P_X_te_conf.npy'
    with open(writepath, 'wb') as f:
        np.save(f,P_X_te_conf)
    writepath = '/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/results_notshuffled_Q_X_te_conf.npy'
    with open(writepath, 'wb') as f:
        np.save(f,Q_X_te_conf)
    writepath = '/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/results_notshuffled_QQ_X_te_conf.npy'
    with open(writepath, 'wb') as f:
        np.save(f,QQ_X_te_conf)

    writepath = '/content/gdrive/MyDrive/MIT/9.66/assemblies_mnist_demo/results_notshuffled_test_data.npy'
    with open(writepath, 'wb') as g:
        np.save(g, X_te)


    # area1 = Area(n,k,p)
    # P_X_tr = np.vstack([area1.project(X_tr[i,:],20)[np.newaxis,:] for i in range(2*N)])
    # P_X_te = np.vstack([area1.project(X_te[i,:],20)[np.newaxis,:] for i in range(2*N)])
    #
    # clf = LogisticRegression()
    # clf.fit(P_X_tr, labels_tr)
    # print("Test score for project assemblies created randomly: %.3f"%(clf.score(P_X_te,labels_te)))

    # area2 = Area(n,k,p)
    # area2.project_learning_classes(X_tr, 20)
    # Q_X_tr = np.vstack([area2.project(X_tr[i,:],10, update=False, refresh=False)[np.newaxis,:] for i in range(2*N)])
    # Q_X_te = np.vstack([area2.project(X_te[i,:],10, update=False, refresh=False)[np.newaxis,:] for i in range(2*N)])
    #
    # clf = LogisticRegression()
    # clf.fit(Q_X_tr, labels_tr)
    # print("Test score for project assemblies created using learning algorithm: %.3f"%(clf.score(Q_X_te,labels_te)))

    # area3 = Area(n,k,p)
    # RP_Y_tr, RP_Z_tr = [],[]
    # RP_Y_te, RP_Z_te = [],[]
    # for i in range(2*N):
    #     y_tr,z_tr = area3.reciprocal_project(X_tr[i,:],40)
    #     RP_Y_tr.append(y_tr[np.newaxis,:])
    #     RP_Z_tr.append(z_tr[np.newaxis,:])
    #
    #     y_te,z_te = area3.reciprocal_project(X_te[i,:],40)
    #     RP_Y_te.append(y_te[np.newaxis,:])
    #     RP_Z_te.append(z_te[np.newaxis,:])
    #
    # RP_Y_tr = np.vstack(RP_Y_tr)
    # RP_Y_te = np.vstack(RP_Y_te)
    # RP_Z_tr = np.vstack(RP_Z_tr)
    # RP_Z_te = np.vstack(RP_Z_te)
    #
    # clf = LogisticRegression()
    # clf.fit(RP_Y_tr, labels_tr)
    # print("Test score for reciprocal_project assemblies (y) created randomly: %.3f"%(clf.score(RP_Y_te,labels_te)))
    #
    # clf = LogisticRegression()
    # clf.fit(RP_Z_tr, labels_tr)
    # print("Test score for reciprocal_project assemblies (z) created randomly: %.3f"%(clf.score(RP_Z_te,labels_te)))
    #
    # area4 = Area(n,k,p)
    # area4.reciprocal_project_learning_classes(X_tr, 40)
    # RQ_Y_tr, RQ_Z_tr = [],[]
    # RQ_Y_te, RQ_Z_te = [],[]
    # for i in range(2*N):
    #     y_tr,z_tr = area4.reciprocal_project(X_tr[i,:],10, update=False, refresh=False)
    #     RQ_Y_tr.append(y_tr[np.newaxis,:])
    #     RQ_Z_tr.append(z_tr[np.newaxis,:])
    #
    #     y_te,z_te = area4.reciprocal_project(X_te[i,:],10, update=False, refresh=False)
    #     RQ_Y_te.append(y_te[np.newaxis,:])
    #     RQ_Z_te.append(z_te[np.newaxis,:])
    #
    # RQ_Y_tr = np.vstack(RQ_Y_tr)
    # RQ_Y_te = np.vstack(RQ_Y_te)
    # RQ_Z_tr = np.vstack(RQ_Z_tr)
    # RQ_Z_te = np.vstack(RQ_Z_te)
    #
    # clf = LogisticRegression()
    # clf.fit(RQ_Y_tr, labels_tr)
    # print("Test score for reciprocal_project assemblies (y) created using learning algorithm: %.3f"%(clf.score(RQ_Y_te,labels_te)))
    #
    # clf = LogisticRegression()
    # clf.fit(RQ_Z_tr, labels_tr)
    # print("Test score for reciprocal_project assemblies (z) created using learning algorithm: %.3f"%(clf.score(RQ_Z_te,labels_te)))

    # area5 = Area(n,k,p)
    # area5.project_learning_classes(X_tr, 20)
    # Q_X_tr = np.vstack([area5.project(X_tr[i,:],10, update=False, refresh=False)[np.newaxis,:] for i in range(2*N)])
    # Q_X_te = np.vstack([area5.project(X_te[i,:],10, update=False, refresh=False)[np.newaxis,:] for i in range(2*N)])
    #
    # area6 = Area(n,k,p)
    # area6.project_learning_classes(Q_X_tr, 40)
    # QQ_X_tr = np.vstack([area6.project(Q_X_tr[i,:],20, update=False, refresh=False)[np.newaxis,:] for i in range(2*N)])
    # QQ_X_te = np.vstack([area6.project(Q_X_te[i,:],20, update=False, refresh=False)[np.newaxis,:] for i in range(2*N)])
    #
    # clf = LogisticRegression()
    # clf.fit(QQ_X_tr, labels_tr)
    # print("Test score for project assemblies created using double projection learning algorithm: %.3f"%(clf.score(QQ_X_te,labels_te)))
