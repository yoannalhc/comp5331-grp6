import pandas as pd
import numpy as np
from os.path import join


def process_birch(dataset_path, save_path, ds_name):
    data = pd.read_csv(dataset_path, delim_whitespace=True, header=None)
    dim = int(ds_name[3:])

    data.columns = [f'x_{i+1}' for i in range(dim)]

    mean = 0.5
    std_dev = 0.5

    noise_columns = {}
    for i in range(dim):
        noise = np.random.normal(mean, std_dev, size=data.shape[0])
        noise_columns[f'noise_x_{i+1}'] = data[f'x_{i+1}'] + noise

    new_columns = pd.DataFrame(noise_columns)

    data = pd.concat([data, new_columns], axis=1)

    data.to_csv(join(save_path, f'{ds_name}_epsilon.csv'), index=False)

if __name__ == "__main__":
    ds_name = "dim128"
    ds_folder = "high_dim"

    ds_path = f"../../dataset/high_dim/{ds_name}.txt"
    save_path = "../../dataset/high_dim"
    process_birch(ds_path, save_path, ds_name)