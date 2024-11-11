import pandas as pd
import numpy as np
from helper import geo_to_cartesian
from os.path import join, isfile, isdir
from os import mkdir
import datetime

def filter_geo(dataset_path, save_path, ds_name):
    print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: filtering')

    if not isfile(join(save_path, f'{ds_name}_first_week.csv')):
        ds = pd.read_csv(dataset_path, sep="\t", header=None)
        ds.columns = ["user", "check-in time", "latitude", "longitude", "location id"]
        ds = ds.dropna()
        first_week = ds.loc[ds['check-in time'].str.contains(r'2010-01-0[1-7]', regex=True)]
        second_week = ds.loc[ds['check-in time'].str.contains(r'2010-01-(?:0[8-9]|1[0-4])', regex=True)]
        if not isdir(save_path):
           mkdir(save_path)
        first_week.to_csv(join(save_path, f'{ds_name}_first_week.csv'), index=False)
        second_week.to_csv(join(save_path, f'{ds_name}_second_week.csv'), index=False)
    else:
        first_week = pd.read_csv(join(save_path, f'{ds_name}_first_week.csv'))
        second_week = pd.read_csv(join(save_path, f'{ds_name}_second_week.csv'))

    first_week_geo = [geo_to_cartesian(lat, lon) for (lat, lon) in
                      zip(first_week['latitude'], first_week['longitude'])]  # [(x, y, z)]
    first_week_geo = np.asarray(list(map(list, first_week_geo)))  # [[x], [y], [z]]
    second_week_geo = [geo_to_cartesian(lat, lon) for (lat, lon) in
                       zip(second_week['latitude'], second_week['longitude'])]
    second_week_geo = np.asarray(list(map(list, second_week_geo)))
    # print(len(first_week), first_week_geo.shape, first_week_geo)
    # print(len(second_week), second_week_geo.shape, second_week_geo)
    # print(len(first_week), first_week_geo.shape)
    # print(len(second_week), second_week_geo.shape)
    #pd.DataFrame(first_week_geo, columns=['x','y','z']).to_csv(join(save_path, f'{ds_name}_first_week_geo.csv'), index=False)
    #pd.DataFrame(second_week_geo, columns=['x', 'y', 'z']).to_csv(join(save_path, f'{ds_name}_second_week_geo.csv'), index=False)
    first_week_filtered = first_week[["user", "check-in time"]].copy()
    second_week_filtered = second_week[["user", "check-in time"]].copy()
    for i, coor in enumerate(first_week_geo):
        for j, axis in enumerate(('x', 'y', 'z')):
            first_week_filtered.loc[i, axis] = coor[j]
    for i, coor in enumerate(second_week_geo):
        for j, axis in enumerate(('x', 'y', 'z')):
            second_week_filtered.loc[i, axis] = coor[j]

    first_week_filtered.to_csv(join(save_path, f'{ds_name}_first_week_filtered.csv'), index=False)
    second_week_filtered.to_csv(join(save_path, f'{ds_name}_second_week_filtered.csv'), index=False)

    return first_week_filtered, second_week_filtered

def process_geo(dataset_path, save_path, ds_name):
    process_path = join(save_path, 'process')
    if not isfile(join(process_path, f'{ds_name}_first_week_filtered.csv')):
        first_week, second_week = filter_geo(dataset_path, process_path, ds_name)
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

def further_filter_geo(dataset_path, save_path, ds_name):
    print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: further filtering')

    df = pd.read_csv(join(dataset_path, f"{ds_name}_epsilon.csv"), index_col=None)
    x1 = df['x_1'].tolist()
    y1 = df['y_1'].tolist()
    z1 = df['z_1'].tolist()
    x2 = df['x_2'].tolist()
    y2 = df['y_2'].tolist()
    z2 = df['z_2'].tolist()
    user = df['user'].tolist()
    first_week = np.asarray(list(map(list, zip(x1, y1, z1))))
    second_week = np.asarray(list(map(list, zip(x2, y2, z2))))
    filtered = []
    for i, (week_1, week_2) in enumerate(zip(first_week, second_week)):
        dist = np.linalg.norm(week_1 - week_2)
        print(f"dist for {user[i]}: {dist}")
        if dist < 1:
            filtered.append([user[i], *week_1, *week_2])

    filtered = np.asarray(filtered)
    filered_df = pd.DataFrame({'user': filtered[:, 0].astype(int), 'x_1': filtered[:, 1], 'y_1': filtered[:, 2], 'z_1': filtered[:, 3],
                               'x_2': filtered[:, 4], 'y_2': filtered[:, 5], 'z_2': filtered[:, 6]})
    filered_df.to_csv(join(save_path, f'{ds_name}_filtered_epsilon.csv'), index=False)

if __name__ == "__main__":
    ds_names = ["Brightkite", "Gowalla"]
    for ds_name in ds_names:
        #ds_path = f"../../dataset/snap_standford/{ds_name}_totalCheckins.txt"
        #save_path = "../../dataset/snap_standford/"
        #process_geo(ds_path, save_path, ds_name)
        further_filter_geo("../../dataset/snap_standford/", "../../dataset/snap_standford/", ds_name)