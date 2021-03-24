import scipy.io as sci
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import metrics
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from prettytable import PrettyTable
from SMILE import SMILE
from CSFS import CSFS


def main():
    bibtex = sci.loadmat('D:\课程作业\机器学习\机器学习课程设计\dataset\\bibtex.mat')
    medical = sci.loadmat('D:\课程作业\机器学习\机器学习课程设计\dataset\medical.mat')
    bib_X = bibtex['data']          #7395,1836
    bib_y = bibtex['target']        #159,7395
    med_X = medical['data']         #978,1449
    med_y = medical['target']       #45,978
    scaler = MinMaxScaler()
    scaler.fit(bib_X)
    bib_X = scaler.transform(bib_X)
    scaler = MinMaxScaler()
    scaler.fit(med_X)
    med_X = scaler.transform(med_X)

    f1_scores = []
    l2_s = ['l1','l2']
    for l2 in l2_s:
        clf = BinaryRelevance(LogisticRegression(penalty=l2,solver='liblinear',dual=False))
        clf.fit(med_X, med_y.T)
        pre = clf.predict(med_X)
        f1_scores.append(metrics.f1_score(med_y.T, pre, average='samples'))
    for l2 in l2_s:
        clf = BinaryRelevance(LinearSVC(penalty=l2,dual=False))
        clf.fit(med_X, med_y.T)
        pre = clf.predict(med_X)
        f1_scores.append(metrics.f1_score(med_y.T, pre, average='samples'))
    tabel = PrettyTable(["","log","hinge"])
    tabel.padding_width=1
    tabel.add_row(["l1",f1_scores[0],f1_scores[2]])
    tabel.add_row(["l2", f1_scores[1],f1_scores[3]])
    csfs = CSFS(u=0.1)
    W,b=csfs.fit(med_X.T,med_y.T,u=0.1)
    pred=csfs.predict(med_X.T,W,b)
    new_y = np.zeros(med_y.shape)
    size = int(med_y.shape[1]*0.7)
    new_y[:,:size] = med_y[:,:size]
    smile = SMILE(alpha=0.1)
    smile.fit(med_X.T,new_y)
    pred_s=smile.predict(med_X.T)
    csfs2 = CSFS(u=0.1)
    W,b=csfs.fit(med_X.T,new_y.T,u=0.1)
    pred=csfs.predict(med_X.T,W,b)
    print('large mult_score:',metrics.f1_score(med_y.T,pred,average='samples'))
    print('CSFSf1_scores:',metrics.f1_score(med_y.T,pred_s,average='samples'))
    print('SMILE_score:',metrics.f1_score(med_y.T,pred,average='samples'))
    print(tabel)

if __name__=='__main__':
    main()