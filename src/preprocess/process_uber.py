import pandas as pd
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import datetime
import numpy as np
from os.path import join, isfile

def draw(G, first_day_geo, second_day_geo):
    pos = dict()
    pos.update((n, (1, i)) for i, n in enumerate(first_day_geo))
    pos.update((n, (2, i)) for i, n in enumerate(second_day_geo))
    nx.draw(G, with_labels=True, pos=pos)
    plt.show()

def geo_to_cartesian(lat, lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371  # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return x, y, z

def create_graph(first_day, second_day, save_path):
    print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: creating graph')
    first_day_date = first_day['Date/Time']
    second_day_date = second_day['Date/Time']
    first_day_geo = [geo_to_cartesian(lat, lon) for (lat, lon) in zip(first_day['Lat'], first_day['Lon'])]
    second_day_geo = [geo_to_cartesian(lat, lon) for (lat, lon) in zip(second_day['Lat'], second_day['Lon'])]
    first_day_info = {date: np.asarray(coor) for (date, coor) in zip(first_day_date, first_day_geo)}
    second_day_info = {date: np.asarray(coor) for (date, coor) in zip(second_day_date, second_day_geo)}

    G = nx.Graph()
    # use date as node name (unique)
    G.add_nodes_from(first_day_date, bipartite=0)
    G.add_nodes_from(second_day_date, bipartite=1)
    for i, i_coor in first_day_info.items():
        for j, j_coor in second_day_info.items():
            # each node in first day can link to every node in second day, with distance as weight
            dist = np.linalg.norm(i_coor - j_coor)
            #print(f"dist between {i}: {i_coor} and {j}: {j_coor}: {dist}")
            G.add_edge(i, j, weight=dist)

    with open(join(save_path, r"uber_weighted_graph.pickle"), "wb") as output_file:
        pickle.dump(G, output_file)

    with open(join(save_path, r"uber_weighted_graph_info.pickle"), "wb") as output_file:
        pickle.dump((first_day_info, second_day_info), output_file)

    return G, first_day_info, second_day_info

def match_graph(G, save_path):
    print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: matching')
    match = nx.bipartite.minimum_weight_full_matching(G)

    with open(join(save_path, r"uber_weighted_match.pickle"), "wb") as output_file:
        pickle.dump(match, output_file)
    return match

def filter(match, first_day_info, second_day_info, save_path):
    filtered = []
    for date_1, date_2 in match.items():
        if '6/2/2014' in date_1: # don't repeat for second day
            continue
        if np.linalg.norm(first_day_info[date_1] - second_day_info[date_2]) < 1:
            filtered.append([date_1, date_2, *first_day_info[date_1], *second_day_info[date_2]])
    #print("before:", len(filtered), len(filtered[0]))
    df = pd.DataFrame(filtered, columns=['Date/Time1', 'Date/Time2', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2'])
    df.to_csv(join(save_path, r"uber_epsilon.csv"), index=False)

def process_uber(dataset_path, save_path):
    ds = pd.read_csv(dataset_path)
    first_day = ds.loc[ds['Date/Time'].str.contains('6/1/2014')]
    second_day = ds.loc[ds['Date/Time'].str.contains('6/2/2014')]

    if not isfile(join(save_path, r"uber_weighted_match.pickle")):
        if not isfile(join(save_path, "uber_weighted_graph.pickle")):
            G, first_day_info, second_day_info = create_graph(first_day, second_day, save_path)
        else:
            with open(join(save_path, r"uber_weighted_graph.pickle"), "rb") as input_file:
                G = pickle.load(input_file)
            with open(join(save_path, r"uber_weighted_graph_info.pickle"), "rb") as input_file:
                first_day_info, second_day_info = pickle.load(input_file)

        match = match_graph(G, save_path)
    else:
        with open(join(save_path, r"uber_weighted_match.pickle"), "rb") as input_file:
            match = pickle.load(input_file)
        with open(join(save_path, r"uber_weighted_graph_info.pickle"), "rb") as input_file:
            first_day_info, second_day_info = pickle.load(input_file)
    filter(match, first_day_info, second_day_info)

    print(f'{datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}: finish')

if __name__ == "__main__":
    ds_path = "../../dataset/uber/uber-raw-data-jun14.csv"
    save_path = "../../dataset/uber/"
    process_uber(ds_path, save_path)