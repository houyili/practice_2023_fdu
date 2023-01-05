import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import manifold
from time import time

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from pratice_2023_fdu.code.practice_2 import read_orl_image, split_train_test_orl, ORL_SAMPLE_NUM, DEFAULT_SEP

from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform

def plot_lle(fea, label, n_neighbors, n_components, output_path):
    if os.path.exists(output_path + 'practice_3_lle_plot.png'):
        return
    t0 = time()
    Y = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components,
                                        eigen_solver='auto', method='standard').fit_transform(fea)
    t1 = time()
    print("%s: %.2g sec" % ('standard', t1 - t0))
    for i in range(1, 4):
        plt.figure()
        plt.scatter(Y[:100*i, 0], Y[:100*i, 1], c=label[:100*i], cmap=plt.cm.Spectral)
        plt.title("%s %d (%.2g sec)" % ('LLE', n_neighbors , t1 - t0))
        plt.axis('tight')
        plt.savefig(output_path + 'practice_3_lle_plot_%d.png' %i, dpi=500, bbox_inches='tight')

def lle_svm_cls_error(all, train_x, train_y, test_x, test_y, output_path):
    if os.path.exists(output_path + 'practice_3_lle_svm.png'):
        return
    components = [i for i in range(10, 100, 5)]
    res = list()
    handel = open(output_path + 'practice_3_lle_svm_detail_results.txt', 'w')

    for i in components:
        model = manifold.LocallyLinearEmbedding(n_neighbors=ORL_SAMPLE_NUM, n_components=i, method='standard')
        model.fit(all)
        x_pca = model.transform(train_x)
        t_pca = model.transform(test_x)
        param_grid = {
            "C": loguniform(1e3, 1e5),
            "gamma": loguniform(1e-4, 1e-1),
        }
        smv = RandomizedSearchCV(
            SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10
        )
        smv.fit(x_pca, train_y)
        pred = smv.predict(t_pca)
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
    plt.plot(components, res, label="SVM with LLE", marker='o')
    plt.xlabel("Component Nums.")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig(output_path + 'practice_3_lle_svm.png', dpi=500, bbox_inches='tight')

def lle_lda_cls_error(all, train_x, train_y, test_x, test_y, output_path):
    if os.path.exists(output_path + 'practice_3_lle_lda.png'):
        return
    components = [i for i in range(10, 100, 5)]
    res = list()
    handel = open(output_path + 'practice_3_lle_lda_detail_results.txt', 'w')

    for i in components:
        model = manifold.LocallyLinearEmbedding(n_neighbors=ORL_SAMPLE_NUM, n_components=i, method='standard')
        model.fit(all)
        x_pca = model.transform(train_x)
        t_pca = model.transform(test_x)
        clf = LinearDiscriminantAnalysis()
        clf.fit(x_pca, train_y)
        pred = clf.predict(t_pca)
        acc = metrics.accuracy_score(test_y, pred)
        print("PCA component num %d, acc is: %0.4f" % (i, acc))
        res.append(acc)
        if acc > 0.90 and not handel.closed:
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
    plt.plot(components, res, label="LDA with LLE", marker='o')
    plt.xlabel("Component Nums.")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig(output_path + 'practice_3_lle_lda.png', dpi=500, bbox_inches='tight')


def lle_lr_cls_error(all, train_x, train_y, test_x, test_y, output_path):
    if os.path.exists(output_path + 'practice_3_lle_lr.png'):
        return
    components = [i for i in range(10, 200, 5)]
    res = list()
    handel = open(output_path + 'practice_3_lle_lr_detail_results.txt', 'w')

    for i in components:
        model = manifold.LocallyLinearEmbedding(n_neighbors=ORL_SAMPLE_NUM, n_components=i, method='standard')
        model.fit(all)
        x_pca = model.transform(train_x)
        t_pca = model.transform(test_x)
        clf = LogisticRegression(max_iter=200)
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
    plt.plot(components, res, label="Logistic Regression with LLE", marker='o')
    plt.xlabel("Component Nums.")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.legend(loc=4)
    plt.savefig(output_path + 'practice_3_lle_lr.png', dpi=500, bbox_inches='tight')


if __name__ == '__main__':
    image_file_path = "../data/ORL"
    out_file_path = "../data/ORL/"
    out_res_path = "../results/"
    fea, label = read_orl_image(image_file_path, out_file_path)
    train_x, train_y, test_x, test_y = split_train_test_orl(fea, label)
    plot_lle(fea, label, ORL_SAMPLE_NUM, 2, out_res_path)
    lle_svm_cls_error(fea, train_x, train_y, test_x, test_y, out_res_path)
    lle_lda_cls_error(fea, train_x, train_y, test_x, test_y, out_res_path)
    lle_lr_cls_error(fea, train_x, train_y, test_x, test_y, out_res_path)