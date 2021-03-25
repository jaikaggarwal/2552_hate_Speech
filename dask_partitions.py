import dask.dataframe as dd 
import time
import tqdm

OUTPUT_DIR = '/ais/hal9000/jai/autism/2552_partitions/'
df = dd.read_json('/ais/hal9000/jai/autism/2552_processed/RC_2017-08.txt', lines=True, blocksize=2**28)
print("READING DONE")
a = time.time()
for i in tqdm.tqdm(range(df.npartitions)):
    v1 =  df.partitions[i].groupby('subreddit')
    b = time.time()
    print("V1 DONE")
    print((b-a)/60)
    v2 = v1.count().compute()
    print("V2 DONE")
    print(v2)
    print((time.time() - b)/60)