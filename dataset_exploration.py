import os
import pandas as pd
import numpy as np 
import tqdm

preordering = ['author', 'body', 'controversiality', 'subreddit', 'created_utc']
ordering = ['body', 'controversiality', 'created_utc']
INPUT_DIR = '/ais/hal9000/jai/autism/2552_partitions/'
dtypes = {'author': 'str', 'subreddit': 'str', 'created_utc': 'int64', 'controversiality': 'int64', 'body': 'str'}

OVERALL_DATA = 'overall_data_counts.csv'
INCEL_DATA = 'incel_data_counts.csv'

def get_overall_user_activity():
    if os.path.isfile(INPUT_DIR + OVERALL_DATA):
        return pd.read_csv(INPUT_DIR + OVERALL_DATA)
    files = os.listdir(INPUT_DIR)
    files = [filename for filename in files if 'INCEL_DATA' not in filename]
    base = pd.read_csv(INPUT_DIR + files[0]).dropna().set_index(['author', 'subreddit'])[ordering]
    
    #add all new files to base
    for filename in tqdm.tqdm(files[1:]):
        base = base.add(pd.read_csv(INPUT_DIR + filename).dropna().set_index(['author', 'subreddit'])[ordering], fill_value=0)
    print(base.shape)
    base.to_csv(INPUT_DIR + OVERALL_DATA)
    return base

def get_incel_user_activity():
    # if os.path.isfile(INPUT_DIR + INCEL_DATA):
    #     return pd.read_csv(INPUT_DIR + INCEL_DATA)
    files = os.listdir(INPUT_DIR)
    files = [filename for filename in files if 'INCEL_DATA' in filename]
    base = pd.read_csv(INPUT_DIR + files[0]).dropna()[preordering].astype(dtypes).groupby(['author', 'subreddit']).nunique()[ordering]

    for filename in tqdm.tqdm(files[1:]):
        print(filename)
        try:
            base = base.add(pd.read_csv(INPUT_DIR + filename).dropna()[preordering].astype(dtypes).groupby(['author', 'subreddit']).nunique()[ordering], fill_value=0)
        except Exception as e:
            print(e)
    print(base.shape)
    base.to_csv(INPUT_DIR + INCEL_DATA)
    return base

def get_control_group(overall_df, incel_df):
    incel_authors = set(incel_df.index)
    print("BEFORE TRIMMING")
    print(overall_df.shape)
    overall_df = overall_df[~overall_df.index.isin(incel_authors)]
    print("AFTER TRIMMING")
    print(overall_df.shape)





if __name__ == "__main__":
    # print("HI")
    # overall_df = get_overall_user_activity()
    incel_df = get_incel_user_activity()
    # get_control_group(overall_df, incel_df)
    


