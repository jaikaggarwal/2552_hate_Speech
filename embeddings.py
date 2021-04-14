import os
import pandas as pd
import numpy as np 
import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

ordering = ['body', 'controversiality', 'subreddit', 'created_utc']
INPUT_DIR = '/ais/hal9000/jai/autism/2552_partitions/'
OUTPUT_DIR = '/ais/hal9000/jai/autism/'
dtypes = {'author': 'str', 'subreddit': 'str', 'created_utc': 'int64', 'controversiality': 'int64', 'body': 'str'}

OVERALL_DATA = 'overall_data_counts.csv'
INCEL_DATA = 'incel_data_counts.csv'
EMBEDDINGS = '~/AutismMarkers/data_files/reddit-master-vectors.tsv'
EMBEDDING_METADATA = '~/AutismMarkers/data_files/reddit-master-metadata.tsv'
MANOSPHERE = ['mgtow', 'braincels', 'incels', 'pussypass', 'theredpill']
COMMENT_THRESHOLD = 100
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

def preprocess_counts(true_communities, use_prior=False):
    if use_prior and os.path.isfile(OUTPUT_DIR + 'thresholded_data.csv'):
        return pd.read_csv(OUTPUT_DIR + 'thresholded_data.csv').set_index('author')
    df = pd.read_csv(INPUT_DIR + OVERALL_DATA).set_index(['author', 'subreddit'])
    attested_communities = df.index.get_level_values('subreddit').unique()
    aberrant_communities = set(attested_communities).difference(set(true_communities))
    new_df = df.drop(aberrant_communities, level='subreddit')
    smaller_df = new_df.groupby('author').sum()
    threshold_df = smaller_df[smaller_df['body'] > COMMENT_THRESHOLD]
    final_df = new_df[new_df.index.isin(threshold_df.index, level='author')]
    final_df.to_csv(OUTPUT_DIR + 'thresholded_data.csv')
    return final_df

def initialize_all_user_embeddings():
    #TODO: SHOULD NOT NEED THIS LINE BELOW
    overall_counts = preprocess_counts(list(sub_to_idx.keys()), True)
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

def rq2():
    control = pd.read_csv('rq2.control.csv').set_index(['author', 'subreddit'])
    treatment = pd.read_csv('rq2.treatment.csv').set_index(['author', 'subreddit'])


    unique_control = control.index.get_level_values('author').unique().tolist()
    unique_treatment = treatment.index.get_level_values('author').unique().tolist()

    control_author_to_GS = {}
    for author in tqdm.tqdm(unique_control):
        sub_df = control.loc[author]
        sub_counts = sub_df['body'].to_dict()
        user_embedding = get_user_embedding(sub_counts)
        gs_score = generalist_specialist_score(user_embedding, sub_counts)
        control_author_to_GS[author] = gs_score

    treatment_author_to_GS = {}
    for author in tqdm.tqdm(unique_treatment):
        sub_df = treatment.loc[author]
        sub_counts = sub_df['body'].to_dict()
        user_embedding = get_user_embedding(sub_counts)
        gs_score = generalist_specialist_score(user_embedding, sub_counts)
        treatment_author_to_GS[author] = gs_score

    control_user_gs_df = pd.DataFrame.from_dict(control_author_to_GS, orient='index')
    control_user_gs_df.to_csv(OUTPUT_DIR + 'gs_scores.control.csv')

    treatment_user_gs_df = pd.DataFrame.from_dict(treatment_author_to_GS, orient='index')
    treatment_user_gs_df.to_csv(OUTPUT_DIR + 'gs_scores.treatment.csv')

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

    return np.sum(weighted_sims) / np.sum(weights)

def valence_control():
    ue_df = pd.read_csv(OUTPUT_DIR + 'all_user_embeddings.csv').rename(columns={'Unnamed: 0': 'author'}).set_index('author')
incel_data = pd.read_csv(INPUT_DIR + INCEL_DATA).set_index(['author', 'subreddit'])
incel_authors = incel_data.index.get_level_values('author').unique().tolist()
    incel_embeddings_df = ue_df[ue_df.index.isin(incel_authors)]
    overall_embeddings_df = ue_df[~ue_df.index.isin(incel_authors)]
    incel_embeddings = incel_embeddings_df.to_numpy()
    overall_embeddings = overall_embeddings_df.to_numpy()

    cos_matrix = cosine_similarity(incel_embeddings, overall_embeddings)
    max_indices = np.argmax(cos_matrix, axis=1)

    assert len(max_indices) == incel_embeddings_df.shape[0]

    matched_control = overall_embeddings_df.iloc[max_indices]
    matched_control.to_csv('rq1.control.csv')
    incel_embeddings_df.to_csv('rq1.treatment.csv')


def gs_control():
    thresholded = pd.read_csv(OUTPUT_DIR + 'thresholded_data.csv').set_index('author').groupby('author').sum()

    incel_data = pd.read_csv(INPUT_DIR + INCEL_DATA).set_index(['author', 'subreddit'])
    incel_authors = incel_data.index.get_level_values('author').unique().tolist()
    incel_data = thresholded[thresholded.index.isin(incel_authors)]
    overall_data = thresholded[~thresholded.index.isin(incel_authors)]

    valid_incel_authors = incel_data.groupby('author').sum().index.tolist()
    overall_authors = overall_data.groupby('author').sum().sample(len(valid_incel_authors)).index.tolist()


    thresholded = pd.read_csv(OUTPUT_DIR + 'thresholded_data.csv').set_index('author')
    treatment = thresholded[thresholded.index.isin(valid_incel_authors)]
    control = thresholded[thresholded.index.isin(overall_authors)]

    treatment.to_csv('rq2.treatment.csv')
    control.to_csv('rq2.control.csv')



def rq2_viz():
control = pd.read_csv(OUTPUT_DIR + 'gs_scores.control.csv')
treatment = pd.read_csv(OUTPUT_DIR + 'gs_scores.treatment.csv')
    plt.hist(control['0'].tolist())
    plt.xlim(0.5, 1.5)
    plt.ylim(0, 1500)
    plt.savefig('control.rq2.png')
    plt.clf()
    plt.hist(treatment['0'].tolist())
    plt.xlim(0.5, 1.5)
    plt.ylim(0, 1500)
    plt.savefig('treatment.rq2.png')
    plt.clf()

    #TODO: CONTROL FOR VOLUME OF ACTIVITY

    #TODO: FIGURE OUT WHY NOT ALL AUTHORS ARE IN THRESHOLDED DATA



if __name__ == "__main__":
    # initialize_all_user_embeddings()
    # valence_control()
    # gs_control()
    rq2_viz()
    # user_embedding_df = main()



# TODO:
# Get control group for GS scores
# Run GS code
# plot distributions
# run two sample KS test

# Load valence code
# Run through whole dataset and find posts from our user set
# Compare the valence scores across the two and get cohen's d

# All of RQ3





    
    

    
    
    