from os.path import join, basename
import numpy as np
import pickle

def find_pair_assign(ds, result_file, eval_path):
    pair1, pair2 = ds.load()

    with open(result_file, 'rb') as input_file:
        center1, cluster1, center2, cluster2 = pickle.load(input_file)
    # print(cluster1)
    #print("center1:", center1, "cluster1:", cluster1, "center2", center2, "cluster2:", cluster2, sep="\n")
    c1_pt = np.asarray([c[0] for c in cluster1])  # [(pt, label)] -> [pt][label]
    c1_label = np.asarray([c[1] for c in cluster1]).flatten()
    # if not isinstance(c1_label, np.int64):
    #     c1_label = c1_label.flatten()
    c2_pt = np.asarray([c[0] for c in cluster2])  # [(pt, label)] -> [pt][label]
    c2_label = np.asarray([c[1] for c in cluster2]).flatten()
    # if not isinstance(c2_label, np.int64):
    #     c2_label = c2_label.flatten()

    medoid1 = []
    medoid2 = []
    # for i in range(3):
    #     print(i,":", cluster1[i])
    #     print(i,":", cluster2[i])
    for p1, p2 in zip(pair1, pair2):
        match_p1 = np.where(np.all(c1_pt == p1, axis=1))
        match_p2 = np.where(np.all(c2_pt == p2, axis=1))
        # print("pair:", p1, p2, "match:", c1_pt[match_p1], c2_pt[match_p2])
        # print("match p1:", match_p1, "match p2:", match_p2)
        old_label = c1_label[match_p1[0]][0]
        new_label = c2_label[match_p2[0]][0]
        #print("old_label:", old_label, "new_label:", new_label)
        old_medoid = center1[old_label]
        new_medoid = center2[new_label]
        medoid1.append(old_medoid)
        medoid2.append(new_medoid)
    medoid1 = np.asarray(medoid1)
    medoid2 = np.asarray(medoid2)
    with open(join(eval_path, f"{basename(result_file).split('.')[0]}_eval.pickle"), 'wb') as output_file:
       pickle.dump((pair1, pair2, medoid1, medoid2), output_file)
    return pair1, pair2, medoid1, medoid2

#if __name__ == '__main__':
    #from src.datasets import Uber
    #ds = Uber("../../dataset/uber/uber_epsilon.csv", k=10, lamb=0)
    #find_pair_assign(ds, result_path="../../results/resilient_k/Uber", eval_path=None, algo="gonz", alpha=0.5, beta=0.5)