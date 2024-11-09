import pandas as pd
import numpy as np

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
    
class Birch(Dataset):
    def __init__(self, path, lamb, k):
        super().__init__(path, lamb, k)
        self.name = 'Birch'
    
    def load(self):
        df = pd.read_csv(self.path, index_col = None)
        x1 = df['x_1'].tolist() 
        y1 = df['y_1'].tolist()
        x2 = df['x_2'].tolist()
        y2 = df['y_2'].tolist()
        pair_1 = np.asarray(list(map(list, zip(x1, y1))))
        pair_2 = np.asarray(list(map(list, zip(x2, y2))))
        return pair_1, pair_2
    
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
            

if __name__ == '__main__':
    # first_day, second_day = Uber("../dataset/uber/uber_epsilon.csv").load()
    # first_day, second_day = Geo("../dataset/snap_standford/brightkite_epsilon.csv", "Brightkite", 1.1, 10).load()
    # first_day, second_day = Birch("../dataset/birch/birch3_epsilon.csv", 1.1, 10).load()
    first_day, second_day = HighDim("../dataset/high_dim/dim128_epsilon.csv", 1.1, 10, 128).load()
    print(len(first_day), len(first_day[0]))