import os
import pandas as pd
import numpy as np 
import tqdm
from sklearn.metrics.pairwise import cosine_similarity


ordering = ['body', 'controversiality', 'subreddit', 'created_utc']
INPUT_DIR = '/ais/hal9000/jai/autism/2552_partitions/'
OUTPUT_DIR = '/ais/hal9000/jai/autism/'
dtypes = {'author': 'str', 'subreddit': 'str', 'created_utc': 'int64', 'controversiality': 'int64', 'body': 'str'}

OVERALL_DATA = 'overall_data_counts.csv'
INCEL_DATA = 'incel_data_counts.csv'
EMBEDDINGS = '~/AutismMarkers/data_files/reddit-master-vectors.tsv'
EMBEDDING_METADATA = '~/AutismMarkers/data_files/reddit-master-metadata.tsv'
MANOSPHERE = ['mgtow', 'braincels', 'incels', 'pussypass', 'theredpill']

def preprocess_community_embeddings():
    metadata = pd.read_csv(EMBEDDING_METADATA, delimiter='\t')
    assert metadata.shape[0] == community_embeddings.shape[0]
    meta_map = metadata['community'].to_dict()
    sub_to_idx = {v: k for (k, v) in meta_map.items()}
    return sub_to_idx
    # Turn into dictionary with a subreddit to embeddings map

community_embeddings = pd.read_csv(EMBEDDINGS, delimiter='\t', header=None)
sub_to_idx = preprocess_community_embeddings()



def get_user_embedding(sub_counts, exclude_manosphere=False):
    # Input is dictionary with counts per subreddit
    # Multiply com_embeddings[key]*value for all (key, values)
    # sum into final vector
    final_embedding = None
    for sub, count in sub_counts.items():
        if exclude_manosphere and (sub.lower() in MANOSPHERE):
            continue
        curr_embedding = count * community_embeddings.iloc[sub_to_idx[sub]].to_numpy()
        if final_embedding is None:
            final_embedding = curr_embedding
        else:
            final_embedding += curr_embedding
    return final_embedding if final_embedding is not None else np.zeros(150)

def preprocess_counts(df, true_communities):
    attested_communities = df.index.get_level_values('subreddit').unique()
    aberrant_communities = set(attested_communities).difference(set(true_communities))
    new_df = df.drop(aberrant_communities, level='subreddit')
    return new_df

def initialize_all_user_embeddings():
    overall_counts = pd.read_csv(INPUT_DIR + OVERALL_DATA).set_index(['author', 'subreddit'])
    #TODO: SHOULD NOT NEED THIS LINE BELOW
    overall_counts = preprocess_counts(overall_counts, list(sub_to_idx.keys()))
    unique_authors = overall_counts.index.get_level_values('author').unique().tolist()
    author_to_embedding = {}
    for author in tqdm.tqdm(unique_authors):
        sub_df = overall_counts.loc[author]
        sub_counts = sub_df['body'].to_dict()
        user_embedding = get_user_embedding(sub_counts, True)
        author_to_embedding[author] = user_embedding
    user_embedding_df = pd.DataFrame.from_dict(author_to_embedding, orient='index')
    user_embedding_df.to_csv(OUTPUT_DIR + 'all_user_embeddings.csv')

def main():
    # if os.path.isfile(OUTPUT_DIR + 'user_embeddings.csv'):
    #     return pd.read_csv(OUTPUT_DIR + 'user_embeddings.csv')
    overall_counts = pd.read_csv(INPUT_DIR + OVERALL_DATA).set_index(['author', 'subreddit'])
    #TODO: SHOULD NOT NEED THIS LINE BELOW
    overall_counts = preprocess_counts(overall_counts, list(sub_to_idx.keys()))
    ############################################################################################################
    unique_authors = overall_counts.index.get_level_values('author').unique().tolist()

    ## TODO: GET RID OF RANDOM SAMPLE HERE
    unique_authors = np.random.choice(unique_authors, 1000, False)
    author_to_embedding = {}
    author_to_GS = {}
    for author in tqdm.tqdm(unique_authors):
        sub_df = overall_counts.loc[author]
        sub_counts = sub_df['body'].to_dict()
        user_embedding = get_user_embedding(sub_counts)
        author_to_embedding[author] = user_embedding
        gs_score = generalist_specialist_score(user_embedding, sub_counts)
        author_to_GS[author] = gs_score

    user_embedding_df = pd.DataFrame.from_dict(author_to_embedding, orient='index')
    user_embedding_df.to_csv(OUTPUT_DIR + 'sample_user_embeddings.csv')

    user_gs_df = pd.DataFrame.from_dict(author_to_GS, orient='index')
    user_gs_df.to_csv(OUTPUT_DIR + 'gs_scores.csv')
    return user_embedding_df

def get_control_group():
    
def generalist_specialist_score(user_embedding, sub_counts):
    keys = list(sub_counts.keys())
    weights = np.array([sub_counts[key] for key in keys]).reshape(-1, 1)
    assert weights.shape == (len(keys), 1)
    idxs = [sub_to_idx[key] for key in keys]
    embeddings = community_embeddings.iloc[idxs].to_numpy()
    assert len(weights) == len(keys)
    assert embeddings.shape[0] == len(keys)
    
    sims = cosine_similarity(user_embedding.reshape(1, -1), embeddings).reshape(-1, 1)
    # print(sub_counts)
    # print(keys)
    # print(weights)
    # print(sims)
    # print(sims.shape)
    # print(embeddings.shape)
    assert sims.shape == (len(keys), 1)
    weighted_sims = np.multiply(sims, weights)

    return np.sum(weighted_sims) / len(weights)




if __name__ == "__main__":
    initialize_all_user_embeddings()
    # user_embedding_df = main()






    
    

    
    
    