import gzip
import os
import json 
import tqdm

INPUT_DIR = '/ais/hal9000/jai/autism/2552_gz/'
OUTPUT_DIR = '/ais/hal9000/jai/autism/2552_processed/'
FIELDS_TO_KEEP = ['controversiality', 'author', 'subreddit', 'body', 'created_utc']
filenames = os.listdir(INPUT_DIR)
filenames = [filename for filename in filenames if '2014' in filename]

for filename in tqdm.tqdm(filenames):
    f = gzip.open(INPUT_DIR + filename)
    with open(OUTPUT_DIR + filename[:-3] + '.txt    ', 'w') as fout:
        for line in tqdm.tqdm(f):
            try:
                data_dict = json.loads(line.decode('utf-8'))
                new_dict = {key: val for key, val in data_dict.items() if key in FIELDS_TO_KEEP}
                fout.write(json.dumps(new_dict))
                fout.write('\n')
            except json.decoder.JSONDecodeError:
                continue
        
        
