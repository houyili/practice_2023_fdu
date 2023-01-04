import  pandas  as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster._kmeans import kmeans_plusplus
from sklearn.metrics import accuracy_score


def open_data(path:str):
    df = pd.read_csv(path, header=None, delimiter='\t')  #
    ar = df.values
    feature = ar[:,:-1]
    label = ar[:,-1]
    print(feature.shape)
    print(label.shape)
    return feature, label.astype('int')

def run_kmeans(features, cluster_num=3, max_iter=100, seed=0):
    centers,_ = kmeans_plusplus(features, cluster_num, random_state=seed)
    k_means = KMeans(n_clusters=cluster_num, init=centers, n_init=1, max_iter=max_iter, random_state=seed).fit(features)
    label = k_means.labels_
    Centroids = k_means.cluster_centers_  # 查看质心
    Inertia = k_means.inertia_  # 每个簇内到其质心的距离和，越小越相似
    return label.astype('int'), centers, Centroids

def max_acc_trans_res(res, label):
    res_d = dict()
    for i in range(len(res)):
        if res[i] not in res_d.keys():
            res_d[res[i]] = [0, 0, 0]
        res_d[res[i]][label[i]] += 1
    res_trans = dict()
    # print(res_d)
    for i in res_d.keys():
        res_trans[i] = res_d[i].index(max(res_d[i], key = abs))
    for i in range(len(res)):
        res[i] = res_trans[res[i]]
    return res_trans

def evel_write_res(pred, init, center, label, file_handel, prefix, cluster_num=3):
    max_acc_trans_res(pred, label)
    acc = accuracy_score(label, pred)
    file_handel.write("%s, the accuracy score is: %0.4f\n" %(prefix, acc))
    file_handel.write("The pred label are: \n")
    pred.tofile(file_handel, sep=',', format='%d')
    file_handel.write("\n")
    file_handel.write("The initial center are: \n")
    for i in range(cluster_num):
        init[:,i].tofile(file_handel, sep=',', format='%0.3f')
        file_handel.write("\t" if i < cluster_num - 1 else "\n")
    file_handel.write("The cluster center are: \n")
    for i in range(cluster_num):
        center[:, i].tofile(file_handel, sep=',', format='%0.3f')
        file_handel.write("\t" if i < cluster_num - 1 else "\n")
    return acc

def run_main(data_file, output, summary):
    time = 10
    fea, label = open_data(data_file)
    handel = open(output, 'w')
    iters = [i for i in range(3, 33 ,3)]
    test_num = 0
    res_dict = dict()
    for iter in iters:
        for i in range(time):
            prefix = "Test %d, and max iterator is %d" %(test_num, iter)
            pred, init, center = run_kmeans(fea, max_iter=iter, seed=test_num)
            acc = evel_write_res(pred, init, center, label, handel, prefix)
            if iter not in res_dict:
                res_dict[iter] = [acc]
            else:
                res_dict[iter].append(acc)
            test_num = test_num + 1
    handel.close()
    handel_s = open(summary, 'w')
    for i in res_dict.keys():
        avg = np.mean(res_dict[i])
        std = np.std(res_dict[i])
        max = np.max(res_dict[i])
        handel_s.write("We choose %d kinds of initial seed under super-parameter <Max Iterations of K-Means = %d>, \n"
                       "The average accuracy : %0.4f, standard deviation : %0.4f, "
                       "max accuracy: %0.4f.\n" %(time, i, avg, std, max))
    handel_s.close()


if __name__ == '__main__':
    data_file = "../data/iris.dat"
    out_file = "../results/practice_1_kmeans_detail.txt"
    summary_file = "../results/practice_1_kmeans_summary.txt"
    run_main(data_file, out_file, summary_file)


