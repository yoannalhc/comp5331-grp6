import pandas as pd
import numpy as np
from os.path import join


def process_birch(dataset_path, save_path, ds_name):
    data = pd.read_csv(dataset_path, delim_whitespace=True, header=None)

    data.columns = ['x_1', 'y_1']

    mean = 0.5
    std_dev = 0.5

    noise_x = np.random.normal(mean, std_dev, size=data.shape[0])
    noise_y = np.random.normal(mean, std_dev, size=data.shape[0])

    data['x_2'] = data['x_1'] + noise_x
    data['y_2'] = data['y_1'] + noise_y

    data.to_csv(join(save_path, f'{ds_name}_epsilon.csv'), index=False)

if __name__ == "__main__":
    ds_name = "birch3"

    ds_path = f"../../dataset/birch/{ds_name}.txt"
    save_path = "../../dataset/birch"
    process_birch(ds_path, save_path, ds_name)