import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, path):
        self.path = path

    def load(self):
        pass

class Uber(Dataset):
    def __init__(self, path):
        super().__init__(path)
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
        second_day= np.asarray(list(map(list, zip(x2, y2, z2))))
        return first_day, second_day

if __name__ == '__main__':
    first_day, second_day = Uber("../dataset/Uber/uber_epsilon.csv").load()
    print(len(first_day), len(first_day[0]))