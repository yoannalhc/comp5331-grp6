import pandas as pd
import numpy as np
from helper import geo_to_cartesian
from os.path import join, isfile, isdir
from os import mkdir
import datetime

def filter_geo(dataset_path, save_path):
    print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: filtering')
    ds = pd.read_csv(dataset_path, sep="\t", header=None)
    ds.columns = ["user", "check-in time", "latitude", "longitude", "location id"]
    ds = ds.dropna()
    first_week = ds.loc[ds['check-in time'].str.contains(r'2010-01-0[1-7]', regex=True)]
    second_week = ds.loc[ds['check-in time'].str.contains(r'2010-01-(?:0[8-9]|1[0-4])', regex=True)]
    if not isdir(save_path):
        mkdir(save_path)
    first_week.to_csv(join(save_path, f'{ds_name}_first_week.csv'), index=False)
    second_week.to_csv(join(save_path, f'{ds_name}_second_week.csv'), index=False)

    first_week_geo = [geo_to_cartesian(lat, lon) for (lat, lon) in
                      zip(first_week['latitude'], first_week['longitude'])]  # [(x, y, z)]
    first_week_geo = np.asarray(list(map(list, first_week_geo)))  # [[x], [y], [z]]
    second_week_geo = [geo_to_cartesian(lat, lon) for (lat, lon) in
                       zip(second_week['latitude'], second_week['longitude'])]
    second_week_geo = np.asarray(list(map(list, second_week_geo)))
    first_week_filtered = first_week[["user", "check-in time"]].copy()
    second_week_filtered = second_week[["user", "check-in time"]].copy()
    for coor1, coor2 in zip(first_week_geo, second_week_geo):
        for i, axis in enumerate(('x', 'y', 'z')):
            first_week_filtered.loc[:, axis] = coor1[i]
            second_week_filtered.loc[:, axis] = coor2[i]
    first_week_filtered.to_csv(join(save_path, f'{ds_name}_first_week_filtered.csv'), index=False)
    second_week_filtered.to_csv(join(save_path, f'{ds_name}_second_week_filtered.csv'), index=False)

    return first_week_filtered, second_week_filtered

def process_geo(dataset_path, save_path, ds_name):
    process_path = join(save_path, 'process')
    if not isfile(join(process_path, f'{ds_name}_first_week_filtered.csv')):
        first_week, second_week = filter_geo(dataset_path, process_path)
    else:
        first_week = pd.read_csv(join(process_path, f'{ds_name}_first_week_filtered.csv'))
        second_week = pd.read_csv(join(process_path, f'{ds_name}_second_week_filtered.csv'))

    print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: matching')
    # find user average x, y, z of each week
    first_week_avg = first_week[['user', 'x', 'y', 'z']].groupby(['user']).mean()
    second_week_avg = second_week[['user', 'x', 'y', 'z']].groupby(['user']).mean()
    #print("fist week avg:", first_week_avg.head())
    #print("second week avg:", second_week_avg.head())
    # filter out users that aren't in both week
    common = pd.merge(first_week_avg, second_week_avg, on="user", suffixes=('_1', '_2'))
    #print(common.head())
    common.to_csv(join(save_path, f'{ds_name}_epsilon.csv'))
    print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: finish')


if __name__ == "__main__":
    #ds_name = "Brightkite"
    ds_name = "Gowalla"
    ds_path = f"../../dataset/snap_standford/{ds_name}_totalCheckins.txt"
    save_path = "../../dataset/snap_standford/"
    process_geo(ds_path, save_path, ds_name)