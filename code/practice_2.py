import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import metrics, neighbors
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform

DEFAULT_SEP = ","
ORL_CLASS_NUM, ORL_SAMPLE_NUM = 40, 10
DEFAULT_TRAIN_PART = [0, 1, 2, 3, 4]
DEFAULT_N_COMPONENTS = 200

def read_orl_image(file_path, ouput_path=None):
    if os.path.exists(ouput_path + 'feature.txt') and os.path.exists(ouput_path + 'label.txt'):
        return read_orl_txt(ouput_path)

    feature, label_list, prefix = np.array([[0]]), [], file_path + "/orl"
    for i in range(ORL_CLASS_NUM):
        for j in range(ORL_SAMPLE_NUM):
            label_list.append(i)
            index = i * ORL_SAMPLE_NUM + j + 1
            data = np.asarray(Image.open(prefix + "%03d.bmp" %index))
            value = np.reshape(data, (1, -1))
            if index == 1:
                feature = value
            else:
                feature = np.row_stack((feature, value))
    label = np.asarray(label_list)
    if ouput_path is not None:
        np.savetxt(ouput_path + 'feature.txt', feature, fmt='%.0f', delimiter=DEFAULT_SEP)
        np.savetxt(ouput_path + 'label.txt', label, fmt='%d', delimiter=DEFAULT_SEP)
    # print(feature.shape) # (400, 10304)
    # print(label.shape) # (400,)
    return feature, label

def read_orl_txt(file_path):
    feature = np.loadtxt(file_path + 'feature.txt', delimiter=DEFAULT_SEP)
    label = np.loadtxt(file_path + 'label.txt')
    # print(feature.shape) # (400, 10304)
    # print(label.shape) # (400,)
    return feature, label

def split_train_test_orl(fea, label):
    train_x, train_y, test_x, test_y = None, None, None, None
    for i in range(ORL_CLASS_NUM):
        for j in range(ORL_SAMPLE_NUM):
            index = i * ORL_SAMPLE_NUM + j
            value = fea[index]
            l = label[index]
            if j in DEFAULT_TRAIN_PART:
                if train_x is None:
                    train_x, train_y = value, l
                else:
                    train_x = np.row_stack((train_x, value))
                    train_y = np.row_stack((train_y, l))
            else:
                if test_x is None:
                    test_x, test_y = value, l
                else:
                    test_x = np.row_stack((test_x, value))
                    test_y = np.row_stack((test_y, l))
    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    # print(test_y)
    return train_x, np.reshape(train_y, (-1,)), test_x, np.reshape(test_y, (-1,))

def pca_face(n_components, x):
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True)
    pca.fit(x)
    x_pca = pca.transform(x)
    # print(x_pca.shape)
    return x_pca, pca

def test_pac_recover_error(fea, output_path):
    if os.path.exists(output_path + 'practice_2_pca_recovery_error.png'):
        return
    components, mses = [], []
    for i in range(10, 310, 10):
        x_pca, model = pca_face(i, fea)
        fea_recover = model.inverse_transform(x_pca)
        mse = metrics.mean_squared_error(fea, fea_recover)
        components.append(i)
        mses.append(mse)
    print(components, mses)
    plt.style.use('seaborn-paper')
    plt.figure(1, facecolor="white")
    plt.cla()
    plt.plot(components, mses, label="Mean squared error varying with the number of components of PCA", marker='o')
    plt.xlabel("Component Nums.")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path + 'practice_2_pca_recovery_error.png', dpi=500, bbox_inches='tight')

def knn_cls_error(train_x, train_y, test_x, test_y, output_path):
    if os.path.exists(output_path + 'practice_2_nearest_neighbors_classifier.png'):
        return
    max_d = 6
    degrees = [i for i in range(1, max_d + 1)]
    res = []
    handel = open(output_path + 'practice_2_nearest_neighbors_classifier_detail_results.txt', 'w')
    handel.write("groud_truth_label, prediction\n")
    for weights in ["uniform", "distance"]:
        for k in degrees:
            clf = neighbors.KNeighborsClassifier(k, weights=weights)
            clf.fit(train_x, train_y)
            pred = clf.predict(test_x)
            acc = metrics.accuracy_score(test_y, pred)
            if len(res) == 0:
                to_write = np.column_stack((test_y, pred))
                for i in range(len(test_y)):
                    to_write[i].tofile(handel, sep=DEFAULT_SEP, format="%.0f")
                    handel.write("\n")
                handel.close()
            print("Degree num %d, acc is: %0.4f" %(k, acc))
            res.append(acc)
    plt.style.use('classic')
    plt.figure()
    plt.cla()
    plt.ylim(0.6, 1)
    plt.xlim(1, 6)
    plt.plot(degrees, res[:max_d], label="1-Order distance", marker='o')
    plt.plot(degrees, res[max_d:], label="Euclidean distance", marker='x')
    plt.xlabel("Numbers of Nearest neighbor")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path + 'practice_2_nearest_neighbors_classifier.png', dpi=500, bbox_inches='tight')

def lda_cls_error(all, train_x, train_y, test_x, test_y, output_path):
    if os.path.exists(output_path + 'practice_2_pca_lda.png'):
        return
    components = [i for i in range(10, 35, 5)] + [i for i in range(35, 45)] + [i for i in range(45, 100, 10)]
    res = list()
    handel = open(output_path + 'practice_2_pca_lda_detail_results.txt', 'w')
    handel.write("# The prediction detail Eigenface == N_C \n")
    handel.write("groud_truth_label, prediction\n")
    for i in components:
        _, model = pca_face(i, all)
        x_pca = model.transform(train_x)
        t_pca = model.transform(test_x)
        clf = LinearDiscriminantAnalysis()
        clf.fit(x_pca, train_y)
        pred = clf.predict(t_pca)
        acc = metrics.accuracy_score(test_y, pred)
        print("PCA component num %d, acc is: %0.4f" % (i, acc))
        res.append(acc)
        if i == ORL_CLASS_NUM:
            to_write = np.column_stack((test_y, pred))
            for i in range(len(test_y)):
                to_write[i].tofile(handel, sep=DEFAULT_SEP, format="%.0f")
                handel.write("\n")
            handel.close()
    plt.style.use('classic')
    plt.figure()
    plt.cla()
    plt.xticks(components, components)
    plt.plot(components, res, label="Linear Discriminant Analysis with PCA", marker='o')
    plt.xlabel("Component Nums.")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path + 'practice_2_pca_lda.png', dpi=500, bbox_inches='tight')

def svm_cls_error(all, train_x, train_y, test_x, test_y, output_path):
    if os.path.exists(output_path + 'practice_2_pca_svm.png'):
        return
    components = [i for i in range(10, 301, 5)]
    res = list()
    handel = open(output_path + 'practice_2_pca_svm_detail_results.txt', 'w')

    for i in components:
        _, model = pca_face(i, all)
        x_pca = model.transform(train_x)
        t_pca = model.transform(test_x)
        param_grid = {
            "C": loguniform(1e3, 1e5),
            "gamma": loguniform(1e-4, 1e-1),
        }
        clf = RandomizedSearchCV(
            SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10
        )
        clf.fit(x_pca, train_y)
        pred = clf.predict(t_pca)
        acc = metrics.accuracy_score(test_y, pred)
        print("PCA component num %d, acc is: %0.4f" % (i, acc))
        res.append(acc)
        if acc > 0.88 and not handel.closed:
            handel.write("# PCA component num %d, acc is: %0.4f \n The prediction detail are follows:" % (i, acc))
            handel.write("groud_truth_label, prediction\n")
            print("write")
            to_write = np.column_stack((test_y, pred))
            for i in range(len(test_y)):
                to_write[i].tofile(handel, sep=DEFAULT_SEP, format="%.0f")
                handel.write("\n")
            handel.close()
    plt.style.use('classic')
    plt.figure()
    plt.cla()
    plt.plot(components, res, label="SVM with PCA", marker='o')
    plt.xlabel("Component Nums.")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path + 'practice_2_pca_svm.png', dpi=500, bbox_inches='tight')


if __name__ == '__main__':
    image_file_path = "../data/ORL"
    out_file_path = "../data/ORL/"
    out_res_path = "../results/"
    fea, label = read_orl_image(image_file_path, out_file_path)
    train_x, train_y, test_x, test_y = split_train_test_orl(fea, label)
    test_pac_recover_error(fea, out_res_path)
    knn_cls_error(train_x, train_y, test_x, test_y, out_res_path)
    lda_cls_error(fea, train_x, train_y, test_x, test_y, out_res_path)
    svm_cls_error(fea, train_x, train_y, test_x, test_y, out_res_path)