import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist

class Dataset:
    def __init__(self, path, lamb, k):
        self.path = path
        self.lamb = lamb
        self.k = k

    def load(self):
        pass

class Uber(Dataset):
    def __init__(self, path, lamb, k):
        super().__init__(path, lamb, k)
        self.name = 'Uber'

    def load(self):
        df = pd.read_csv(self.path, index_col=0)
        x1 = df['x1'].tolist()
        y1 = df['y1'].tolist()
        z1 = df['z1'].tolist()
        x2 = df['x2'].tolist()
        y2 = df['y2'].tolist()
        z2 = df['z2'].tolist()
        first_day = np.asarray(list(map(list, zip(x1, y1, z1))))
        second_day = np.asarray(list(map(list, zip(x2, y2, z2))))
        return first_day, second_day

    def get_epsilon(self):
        pair1, pair2 = self.load()
        pair1_pdist = pdist(pair1)
        print(pair1_pdist.shape)
        idx = np.where(pair1_pdist != 0)
        print("idx:", idx[0].shape)
        pair2_pdist = pdist(pair2)
        ratio = pair2_pdist[idx] / pair1_pdist[idx]
        print("divide:", ratio.shape)
        epsilon = np.max(ratio)
        return epsilon

class Geo(Dataset):
    def __init__(self, path, name, lamb, k):
        super().__init__(path, lamb, k)
        self.name = name

    def load(self):
        df = pd.read_csv(self.path, index_col=None)
        x1 = df['x_1'].tolist()
        y1 = df['y_1'].tolist()
        z1 = df['z_1'].tolist()
        x2 = df['x_2'].tolist()
        y2 = df['y_2'].tolist()
        z2 = df['z_2'].tolist()
        first_day = np.asarray(list(map(list, zip(x1, y1, z1))))
        second_day = np.asarray(list(map(list, zip(x2, y2, z2))))
        return first_day, second_day

    def get_epsilon(self):
        pair1, pair2 = self.load()
        pair1_pdist = pdist(pair1)
        print(pair1_pdist.shape)
        idx = np.where(pair1_pdist != 0)
        print("idx:", idx[0].shape)
        pair2_pdist = pdist(pair2)
        ratio = pair2_pdist[idx] / pair1_pdist[idx]
        print("divide:", ratio.shape)
        epsilon = np.max(ratio)
        return epsilon
    
class Birch(Dataset):
    def __init__(self, path, lamb, k, subset):
        super().__init__(path, lamb, k)
        self.name = 'Birch' + str(subset)
        self.subset = subset
    
    def load(self):
        df = pd.read_csv(self.path, index_col = None)
        x1 = df['x_1'].tolist() 
        y1 = df['y_1'].tolist()
        x2 = df['x_2'].tolist()
        y2 = df['y_2'].tolist()
        pair_1 = np.asarray(list(map(list, zip(x1, y1))))
        pair_2 = np.asarray(list(map(list, zip(x2, y2))))
        return pair_1, pair_2

    def get_epsilon(self):
        pair1, pair2 = self.load()
        pair1_pdist = pdist(pair1)
        print(pair1_pdist.shape)
        idx = np.where(pair1_pdist != 0)
        print("idx:", idx[0].shape)
        pair2_pdist = pdist(pair2)
        ratio = pair2_pdist[idx] / pair1_pdist[idx]
        print("divide:", ratio.shape)
        epsilon = np.max(ratio)
        return epsilon
    
class HighDim(Dataset):
    def __init__(self, path, lamb, k, dim):
        super().__init__(path, lamb, k)
        self.name = 'HighDim'+ str(dim)
        self.dim = dim
    
    def load(self):
        df = pd.read_csv(self.path, index_col = None)
        pair_1 = []
        pair_2 = []
        for i in range(self.dim):
            pair_1.append(df[f'x_{i+1}'].tolist())
            pair_2.append(df[f'noise_x_{i+1}'].tolist())
        pair_1 = np.asarray(list(map(list, zip(*pair_1))))
        pair_2 = np.asarray(list(map(list, zip(*pair_2))))
        return pair_1, pair_2

    # def get_epsilon(self):
    #     pair1, pair2 = self.load()
    #     pair1_pdist = pdist(pair1)
    #     print(pair1_pdist.shape)
    #     idx = np.where(pair1_pdist != 0)
    #     print("idx:", idx[0].shape)
    #     pair2_pdist = pdist(pair2)
    #     ratio = pair2_pdist[idx] / pair1_pdist[idx]
    #     print("divide:", ratio.shape)
    #     epsilon = np.max(ratio)
    #     return epsilon
            

if __name__ == '__main__':
    # first_day, second_day = Uber("../dataset/uber/uber_epsilon.csv").load()
    # ds = Uber("../dataset/uber/uber_epsilon.csv", 1.1, 10)
    ds = Geo("../dataset/snap_standford/brightkite_epsilon.csv", "Brightkite", 1.1, 10)
    ep = ds.get_epsilon()
    print("epsilon:", ep)
    # first_day, second_day = Geo("../dataset/snap_standford/brightkite_epsilon.csv", "Brightkite", 1.1, 10).load()
    # first_day, second_day = Birch("../dataset/birch/birch3_epsilon.csv", 1.1, 10).load()
    # first_day, second_day = HighDim("../dataset/high_dim/dim128_epsilon.csv", 1.1, 10, 128).load()
    #print(len(first_day), len(first_day[0]))