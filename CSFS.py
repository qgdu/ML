import numpy as np


class CSFS:

    def __init__(self,u=0.1):

        """Initialize properties.
        :param alpha:
        """
        self.u = u
        self.W = None
        self.H = None
        self.b = None

    def fit(self, data, labels, u):
        '''
        CSFS算法
        :param data:样本X
        :param labels:标签F
        :param u:估计参数
        :return:返回W和预测标签
        '''
        n = len(data[0])  # 样本规格
        c = len(labels[0])  # 标签数
        d = len(data)  # 样本的特征数（有几个特征）
        W = np.random.random(size=(d, c))
        # F = np.zeros((n, c), float)
        l = np.ones((n, 1))
        H = np.eye(n) - 1 / n * l.dot(l.T)
        true_y = labels.copy()
        labeled = list(np.nonzero(np.sum(labels, axis=1))[0])
        labeled_num = len(labeled)
        obji = 1
        iter = 1
        while True:
            D = np.diag(0.5 / (np.linalg.norm(W, axis=1)))
            W = np.linalg.inv(data.dot(H).dot(data.T) + u * D).dot(data).dot(H).dot(labels)
            b = 1 / n * (labels.T.dot(l) - W.T.dot(data).dot(l))
            temp = data.T.dot(W) + l.dot(b.T)
            conver = {}
            for i in range(labeled_num + 1, n):
                for j in range(c):
                    if temp[i][j] > 1:
                        labels[i][j] = 1
                    else:
                        if temp[i][j] < 0:
                            labels[i][j] = 0
                        else:
                            labels[i][j] = temp[i][j]
            conver[iter] = np.linalg.norm((data.T.dot(W) + l.dot(b.T) - labels), 'fro') ** 2 + u * np.sum(np.sqrt(np.sum(np.square(W),axis=1)))
            cver = np.abs((conver[iter] - obji) / obji)
            obji = conver[iter]
            iter += 1
            if cver < 10 ** -3 and iter > 2:
                break
        return W, b

    def predict(self, test, W, b):
        pre_labels = test.T.dot(W) + np.ones((test.shape[1], 1)).dot(b.T)
        for i in range(pre_labels.shape[0]):
            for j in range(pre_labels.shape[1]):
                if pre_labels[i][j] >= 0.5:
                    pre_labels[i][j] = 1
                    continue
                else:
                    pre_labels[i][j] = 0
                    continue
        return pre_labels