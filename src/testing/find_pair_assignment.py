from os.path import join
import numpy as np
import pickle

def find_pair_assign(ds, result_path, eval_path):
    pair1, pair2 = ds.load()

    with open(join(result_path, f"{ds.name}_resilient_{ds.k}.pickle"), 'rb') as input_file:
        center1, cluster1, center2, cluster2 = pickle.load(input_file)
    # print(cluster1)
    c1_pt = np.asarray([c[0] for c in cluster1])  # [(pt, label)] -> [pt][label]
    c1_label = np.asarray([c[1] for c in cluster1]).flatten()
    c2_pt = np.asarray([c[0] for c in cluster2])  # [(pt, label)] -> [pt][label]
    c2_label = np.asarray([c[1] for c in cluster2]).flatten()

    label1 = []
    label2 = []
    # print(c1_label)
    # print(c1_label.shape)
    # print(c1_pt.shape)
    for p1, p2 in zip(pair1, pair2):
        match_p1 = np.where(c1_pt == p1)[0][2]
        match_p2 = np.where(c2_pt == p2)[0][2]
        # print(p1, p2, c1_pt[match_p1], c2_pt[match_p2])
        # print(np.where(c1_pt == p1)[0][2])
        old_label = c1_label[match_p1]
        new_label = c2_label[match_p2]
        # print("old_label:", old_label, "new_label:", new_label)

        # c1_label = center1[c1_label[c1_pt == p1].all()] # find pt label in c1
        # c2_label = center2[c2_label[c2_pt == p2].all()] # find pt label in c2
        label1.append(old_label)
        label2.append(new_label)
    with open(join(eval_path, f"{ds.name}_resilient_{ds.k}_eval.pickle"), 'wb') as output_file:
        pickle.dump((center1, cluster1, center2, cluster2), output_file)
    return pair1, pair2, label1, label2