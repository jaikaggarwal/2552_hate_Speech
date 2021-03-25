import pandas as pd 
import os
import json
import argparse 
import tqdm
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import time

valid_communities = pd.read_csv('../data_files/reddit-master-metadata.tsv', delimiter='\t').community.tolist()
MANOSPHERE = ['mgtow', 'braincels', 'incels', 'pussypass', 'theredpill']
parser = argparse.ArgumentParser()
parser.add_argument("year", help="display a square of a given number",
                    )
parser.add_argument("sector", help="display a square of a given number",
                    )
args = parser.parse_args()

dtypes = {'author': 'str', 'subreddit': 'str', 'created_utc': 'int64', 'controversiality': 'int64', 'body': 'str'}

OUTPUT_DIR = '/ais/hal9000/jai/autism/2552_partitions/'
INPUT_DIR = '/ais/hal9000/jai/autism/2552_processed/'
year = args.year
sector_map = {'first': ['01', '02', '03'], 
              'second': ['04', '05', '06'],
              'third': ['07', '08', '09'],
              'fourth': ['10', '11', '12'] }
months = sector_map[args.sector]


for month in tqdm.tqdm(months):
    df = dd.read_json(f'{INPUT_DIR}RC_{year}-{month}.txt', lines=True, blocksize=2**28, dtype=dtypes)
    # x = df.compute()
    # x.to_csv('test.csv')
    total = None
    print(df.npartitions)
    incel_authors = None
    for i in tqdm.tqdm(range(df.npartitions)):

        # a = time.time()
        # v1 =  df.partitions[i].groupby('author').count().compute()
        # print(v1)
        # b = time.time()
        # print((b-a)/60)
        pdf = df.partitions[i].compute()
        valid_pdf = pdf[pdf['subreddit'].isin(valid_communities)]
        # Get rid of deleted body/comments
        contentful_body_df = valid_pdf[valid_pdf['body'] != '[deleted]'] 
        author_summary_df = contentful_body_df.groupby('author').nunique()
        
        if total is None:
            total = author_summary_df
        else:
            total.add(author_summary_df, fill_value=0)

        pdf['incel'] = pdf['subreddit'].apply(lambda x: x.lower() in MANOSPHERE)
        incel_partition = pdf[pdf['incel']]

        if incel_authors is None:
            incel_authors = incel_partition
        else:
            incel_authors = pd.concat([incel_authors, incel_partition]).reset_index(drop=True)
    total.to_csv(f'{OUTPUT_DIR}RC_{year}-{month}.csv')
    incel_authors.to_csv(f'{OUTPUT_DIR}RC_{year}-{month}_INCEL_DATA.csv')






