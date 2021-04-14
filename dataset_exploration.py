import os
import pandas as pd
import numpy as np 
import tqdm

preordering = ['author', 'body', 'controversiality', 'subreddit', 'created_utc']
ordering = ['body', 'controversiality', 'created_utc']
INPUT_DIR = '/ais/hal9000/jai/autism/2552_partitions/'
dtypes = {'author': 'str', 'subreddit': 'str', 'created_utc': 'int64', 'controversiality': 'int64', 'body': 'str'}

OVERALL_DATA = 'new_overall_data_counts.csv'
INCEL_DATA = 'new_incel_data_counts.csv'

def df_filter(df):
    df = df.dropna()
    sample = df.groupby('author').sample(frac=0.1)
    sample['len'] = sample['body'].apply(lambda x: len(x.split(" ")))
    longer_sample = sample[sample['len'] > 10][preordering]
    return longer_sample

def get_overall_user_activity():
    if os.path.isfile(INPUT_DIR + OVERALL_DATA):
        return pd.read_csv(INPUT_DIR + OVERALL_DATA)
    files = os.listdir(INPUT_DIR)
    files = [filename for filename in files if ('INCEL_DATA' not in filename) and ('rq1' in filename)]
    print(files)
    print(len(files))
    base = df_filter(pd.read_csv(INPUT_DIR + files[0])).set_index(['author', 'subreddit'])[ordering]
    
    #add all new files to base
    for filename in tqdm.tqdm(files[1:]):
        try:
            filtered_df = df_filter(pd.read_csv(INPUT_DIR + filename))
            base = pd.concat([base, filtered_df.set_index(['author', 'subreddit'])[ordering]])
        except:
            continue
        # base = base.add(filtered_df.set_index(['author', 'subreddit'])[ordering], fill_value=0)
    print(base.shape)
    base.to_csv(INPUT_DIR + OVERALL_DATA)
    return base

def get_incel_user_activity():
    if os.path.isfile(INPUT_DIR + INCEL_DATA):
        return pd.read_csv(INPUT_DIR + INCEL_DATA)
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


def data_split():
    data = pd.read_csv(INPUT_DIR + OVERALL_DATA)
    print("INITIAL SHAPE")
    print(data.shape)
    data['author_valid'] = data['author'].apply(lambda x: True if (x not in ["AutoModerator", "[deleted]"]) and (not x.lower().endswith('bot')) else False)
    data = data[data['author_valid']]
    print("NO BOTS OR MODS")
    print(data.shape)
    print("FREQUENT COMMENTORS")
    data_agg = data.groupby('author').count()
    data = data[data['author'].isin(data_agg[data_agg['subreddit'] > 99].index.tolist())]
    print(data.shape)
    incel_data = pd.read_csv(INPUT_DIR + INCEL_DATA).set_index(['author', 'subreddit'])
    incel_authors = incel_data.index.get_level_values('author').unique().tolist()
    control_data = data[~data['author'].isin(incel_authors)]
    incel_data = data[data['author'].isin(incel_authors)]
    print("CONTROL SHAPE")
    print(control_data.shape)
    print("CONTROL AUTHORS")
    print(control_data['author'].nunique())
    print("INCEL SHAPE")
    print(incel_data.shape)
    print("INCEL AUTHORS")
    print(incel_data['author'].nunique())

    control_data.to_csv('rq1.control.v2.csv')
    incel_data.to_csv('rq1.treatment.v2.csv')








if __name__ == "__main__":
    # print("HI")
    # overall_df = get_overall_user_activity()
    # incel_df = get_incel_user_activity()
    data_split()
    # get_control_group(overall_df, incel_df)
    


