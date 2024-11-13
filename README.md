# comp5331-grp6
This repo is for Fall2024 Comp5331 group 6 project: Resilient k-Clustering.

The URL of the GitHub repository is at: https://github.com/yoannalhc/comp5331-grp6

## How to execute 
1. Download the repository to a local environment.
2. Download the datasets and place them in the correct folder. (Refer to [Dataset](#Dataset))
3. Run [COMP5331_Project.ipynb](COMP5331_Project_refactored.ipynb) in order.

## Datasets
Datasets are downloaded from: 
1. BIRCH, HIGH-DIM(low): https://cs.joensuu.fi/sipu/datasets/
2. Uber: https://www.kaggle.com/datasets/fivethirtyeight/uber-pickups-in-new-york-city
3. Brightkite, Gowalla: https://snap.stanford.edu/data/index.html#locnet

Processed datasets can be found at: https://hkustconnect-my.sharepoint.com/:f:/g/personal/hcloaf_connect_ust_hk/En_JVSFmi9hKljeLxGh-nJIBhH5bZDmeVSu16TGjJ7HI7w?e=NLdsAn

Download them and put them into `./dataset`
```
project
📂dataset
└───📂birch
│       │ 📜birch1_epsilon.csv
│       │ 📜birch2_epsilon.csv
│       │ 📜birch3_epsilon.csv
└───📂high_dim
│       │ 📜dim032_epsilon.csv
│       │ 📜dim064_epsilon.csv
│       │ 📜dim128_epsilon.csv
└───📂snap_standford
│       │ 📜Brightkite_epsilon.csv
│       │ 📜Gowalla_epsilon.csv
└───📂uber
        │ 📜uber_epsilon.csv
```

## Description of each source file
- `COMP5331_Project_refactored.ipynb`: The entrance of the program, use it to run
- `src/dataset.py`: Contain dataset classes
- `src/resilient_k.py`: Contain all the resilient algorithm-related classes
- `src/evaluation/py`: Contain all the evaluation-related classes
- `src/preprocess/helper.py`: Contain helper function to process dataset
- `src/preprocess/process_birch.py`: Contain function to process the Birch datasets
- `src/preprocess/process_geo.py`: Contain function to process the geographic datasets
- `src/preprocess/process_high_dim.py`: Contain function to process the high dimensional datasets
- `src/preprocess/process_uber.py`: Contain the function to process the Uber dataset
- `src/testing/find_pair_assignment.py`: Contain the function to find pair assignment in two clustering results
- `src/testing/plot_helper.py`: Contain function to plot the data

## Example
See demo in [COMP5331_Project.ipynb](COMP5331_Project_refactored.ipynb)

## Running Environment
We use Windows OS as the environment in our project.
