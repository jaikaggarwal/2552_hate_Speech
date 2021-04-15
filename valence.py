import re
import sys, csv
from random import shuffle
import numpy as np
from sentence_transformers import SentenceTransformer
import statsmodels.api as sm
from sklearn.metrics import r2_score
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from utils import Serialization
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wilcoxon, ttest_rel
import matplotlib.pyplot as plt
tqdm.pandas()

from sentence_transformers.SentenceTransformer import torch as pt
print(pt.cuda.is_available())
print(pt.__version__)
pt.cuda.set_device(1)
model = SentenceTransformer('bert-large-nli-mean-tokens')

INPUT_DIR = '/ais/hal9000/jai/autism/2552_partitions/'
MANOSPHERE = ['mgtow', 'braincels', 'incels', 'pussypass', 'theredpill']

class LexicalAnalysis():
 
    @staticmethod
    def get_quote_count(post):
        return len(re.findall(r'\bQUOTE\b', post))

    @staticmethod
    def get_link_count(post):
        return len(re.findall(r'\bLINK\b', post))

    @staticmethod
    def get_subreddit_count(post):
        return len(re.findall(r'\bSUBREDDIT\b', post))

    @staticmethod
    def get_TTR(post):
        post = post.lower()
        post = re.sub(r'\W', ' ', post)
        tokens = word_tokenize(post)
        types = nltk.Counter(tokens)
        ttr = len(types)/len(tokens)
        return ttr


    @staticmethod
    def infer_emotion_value(post, regressor_v):
        to_encode = sent_tokenize(post)
        embeddings = model.encode(to_encode)
        v_predictions = regressor_v.predict(embeddings)
        assert(len(v_predictions) == len(embeddings))
        assert type(np.mean(v_predictions)) == np.float64
        return np.mean(v_predictions)


    #### SBERT-related Methods #### 
    @staticmethod
    def fit_beta_reg(y, X, df, group_title):
       
        curr_best_fit = 0
        curr_best_model = None

        for i in tqdm(range(10)):

            X_sample = df.groupby(group_title).apply(lambda temp: temp.sample(int(HELD_OUT_PROP*len(temp))))
            train_idx = pd.Series(X_sample.index.get_level_values(1))
            test_idx = df.index.difference(train_idx).tolist()
            np.random.shuffle(test_idx)
            train_idx = train_idx.tolist()

            # print(train_idx)
            # print(len(train_idx))
            # print(len(test_idx))
            # print(len(set(train_idx).union(set(test_idx))))
            # print(X.shape)
            # print(df.shape)

            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            binom_glm = sm.GLM(y_train, X_train, family=sm.families.Binomial())
            binom_fit_model = binom_glm.fit()
            fit_val = LexicalAnalysis.goodness_of_fit(binom_fit_model, y_test, X_test)

            if fit_val > curr_best_fit:
                print("NEW BEST MODEL")
                print(f"R^2 score of: {fit_val}")
                curr_best_fit = fit_val
                curr_best_model = binom_fit_model
        return curr_best_model
    
    @staticmethod
    def goodness_of_fit(model, true, X):
        y_predicted = model.get_prediction(X)
        pred_vals = y_predicted.summary_frame()['mean']
        fit_val = r2_score(true, pred_vals)
        print(fit_val)
        return fit_val


    @staticmethod
    def none_or_empty(text):
        return text is None or len(text) == 0 or text == "[removed]" or text == '[deleted]'
    
    #### Lexical Analysis Functions ####
    @staticmethod
    def get_embeddings(data, title):
        try:
            embeddings = Serialization.load_obj(title)
        except FileNotFoundError:
            embeddings = model.encode(data, show_progress_bar=True)
            Serialization.save_obj(embeddings, title)
        return embeddings

    @staticmethod
    def init_vad():
        df_vad = pd.read_csv('/ais/hal9000/jai/lexicon.txt', delimiter='\t', header=0)
        df_vad = df_vad.dropna().reset_index(drop=True)
        df = df_vad[['Word', 'Valence']]
        valence = np.array(df['Valence'].tolist())
       
        vad_words = list(df_vad['Word'])

        vad_embeddings = LexicalAnalysis.get_embeddings(vad_words, "vad")

        print("LOADING VALENCE MODEL")
        try:
            valence_model = Serialization.load_obj('valence_model')
        except FileNotFoundError:
            valence_model = LexicalAnalysis.fit_beta_reg(valence, vad_embeddings, df, 'v_group')
            Serialization.save_obj(valence_model, 'valence_model')
        
        LexicalAnalysis.goodness_of_fit(valence_model, valence, vad_embeddings)

        return valence_model

    @staticmethod
    def lexical_metrics(file):
        valence_model = LexicalAnalysis.init_vad()
        # for month in tqdm(months):
            # try:
        data = pd.read_csv(file)
            # except Exception as e:
            #     print(e)
            #     continue
        print(data.shape)
        data = data.dropna()
        data = data[['author', 'body', 'subreddit']]
        print(data.shape)
        data['manosphere'] = data['subreddit'].apply(lambda x: x.lower() in MANOSPHERE)
        data = data[~data['manosphere']]
        print(data.shape)
        data['length'] = data['body'].apply(lambda x: len(x.split(" ")))
        data = data[data['length'] > 10]
        print(data.shape)
        # data = data.sample(100000)
        data['valence'] = data['body'].progress_apply(lambda x: LexicalAnalysis.infer_emotion_value(x, valence_model))

            # for i in tqdm(range(num_data_points)):
            #     post = data.iloc[i]['body']
            #     author = data.iloc[i]['author']
            #     try:
            #         #TODO: BETTER PREPROCESSING
            #         post_valence = LexicalAnalysis.infer_emotion_value(post, valence_model)
            #         if author not in author_to_valence:
            #             author_to_valence[author] = [post_valence]
            #         else:
            #             author_to_valence[author].append(post_valence)
            #     except Exception as e:
            #         print(e)
            #         # all_metric_vals.append([np.nan]*8)
            #         continue
            # user_v_df = pd.DataFrame.from_dict(author_to_valence, orient='index')
        data.to_csv(f'{file[:-4]}_valence_scores.csv')


def data_analysis():
    treatment_1 = pd.read_csv('rq1_treatment_first_valence_scores.csv')
    treatment_2 = pd.read_csv('rq1_treatment_second_valence_scores.csv')
    treatment_3 = pd.read_csv('rq1_treatment_third_valence_scores.csv')
    treatment_4 = pd.read_csv('rq1_treatment_fourth_valence_scores.csv')

    control_1 = pd.read_csv('rq1_control_first_valence_scores.csv')
    control_2 = pd.read_csv('rq1_control_second_valence_scores.csv')
    control_3 = pd.read_csv('rq1_control_third_valence_scores.csv')
    control_4 = pd.read_csv('rq1_control_fourth_valence_scores.csv')

    treatment_data = pd.concat([treatment_1, treatment_2, treatment_3, treatment_4])
    control_data = pd.concat([control_1, control_2, control_3, control_4])

    treatment_mean_valence = treatment_data.groupby('author').mean()
    control_mean_valence = control_data.groupby('author').mean()



    ##### Find matched set
    # Assert that the two sets of authors are mutually exclusive
    # Get the embeddings for each set
    # Run cosine similarity on the two matrices
    # Find the argmax to get most similar
    # Find the number of unique matches
    # Take the set of unique matches and use them as control and treatment respectively (break ties by higher cos sim)

    ue_df = pd.read_csv('/ais/hal9000/jai/autism/all_user_embeddings.csv').rename(columns={'Unnamed: 0': 'author'}).set_index('author')
    treatment_authors = treatment_data['author'].unique().tolist()
    control_authors = control_data['author'].unique().tolist()
    assert len(set(treatment_authors).intersection(control_authors)) == 0
    treatment_embeddings_df = ue_df[ue_df.index.isin(treatment_authors)]
    control_embeddings_df = ue_df[ue_df.index.isin(control_authors)]
    treatment_embeddings = treatment_embeddings_df.to_numpy()
    control_embeddings =  control_embeddings_df.to_numpy()

    cos_matrix = cosine_similarity(treatment_embeddings, control_embeddings)
    max_indices = np.argmax(cos_matrix, axis=1)
    assert len(max_indices) == treatment_embeddings.shape[0]
    most_sim_data  = cos_matrix[np.arange(len(cos_matrix)), max_indices]

    sim_map = {}
    for treatment_idx, control_idx in enumerate(max_indices):
        if control_idx not in sim_map:
            sim_map[control_idx] = treatment_idx
        else:
            if most_sim_data[treatment_idx] > most_sim_data[sim_map[control_idx]]:
                sim_map[control_idx] = treatment_idx

    control_order = []
    treatment_order = []
    for control_auth, treatment_auth in sim_map.items():
        c_auth = control_embeddings_df.iloc[control_auth].name
        t_auth = treatment_embeddings_df.iloc[treatment_auth].name
        control_order.append(c_auth)
        treatment_order.append(t_auth) 

    ordered_control = control_mean_valence.loc[control_order]['valence']
    ordered_treatment = treatment_mean_valence.loc[treatment_order]['valence']

    print(ttest_rel(ordered_control, ordered_treatment))
    cohens_d = (np.mean(ordered_control) - np.mean(ordered_treatment)) / (np.sqrt(((np.std(ordered_control) ** 2 + np.std(ordered_treatment) ** 2) / 2)))
    print(cohens_d)
    print(np.mean(ordered_control))
    print(np.mean(ordered_treatment))

    plt.hist(ordered_control)
    plt.title("Valence Scores of Control Authors")
    plt.xlabel("Valence Scores")
    plt.xlim(0.3, 0.6)
    plt.ylabel("Frequency")
    plt.ylim(0, 1800)
    plt.savefig("rq1.control.v2.png")
    plt.clf()

    plt.hist(ordered_treatment)
    plt.title("Valence Scores of Treatment Authors")
    plt.xlabel("Valence Scores")
    plt.xlim(0.3, 0.6)
    plt.ylabel("Frequency")
    plt.ylim(0, 1800)
    plt.savefig("rq1.treatment.v2.png")
    plt.clf()

    control_contra_df = pd.read_csv('rq1.control.v2.csv')
    treatment_contra_df = pd.read_csv('rq1.treatment.v2.csv')

    control_contra_scores = control_contra_df.groupby('author').mean().loc[control_order]['controversiality']
    treatment_contra_scores = treatment_contra_df.groupby('author').mean().loc[treatment_order]['controversiality']

    print(wilcoxon(control_contra_scores, treatment_contra_scores))
    cohens_d = (np.mean(control_contra_scores) - np.mean(treatment_contra_scores)) / (np.sqrt(((np.std(control_contra_scores) ** 2 + np.std(treatment_contra_scores) ** 2) / 2)))
    print(cohens_d)
    print(np.mean(control_contra_scores))
    print(np.mean(treatment_contra_scores))

    plt.hist(control_contra_scores)
    plt.title("Controversiality Scores of Control Authors")
    plt.xlabel("Controversiality Scores")
    plt.xlim(0, 0.2)
    plt.ylabel("Frequency")
    plt.ylim(0, 1600)
    plt.savefig("rq1.control.v2.contra.png")
    plt.clf()

    plt.hist(treatment_contra_scores)
    plt.title("Controversiality Scores of Treatment Authors")
    plt.xlabel("Controversiality Scores")
    plt.xlim(0, 0.2)
    plt.ylabel("Frequency")
    plt.ylim(0, 1600)
    plt.savefig("rq1.treatment.v2.contra.png")
    plt.clf()




if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("file", help="display a square of a given number",
    #                     )
    # # parser.add_argument("sector", help="display a square of a given number",
    # #                     )
    # args = parser.parse_args()

    # # dtypes = {'author': 'str', 'subreddit': 'str', 'controversiality': 'int64', 'body': 'str'}
    # # year = args.year
    # # sector_map = {'first': ['01', '02', '03'], 
    # #             'second': ['04', '05', '06'],
    # #             'third': ['07', '08', '09'],
    # #             'fourth': ['10', '11', '12'] }
    # # months = sector_map[args.sector]

    # LexicalAnalysis.lexical_metrics(args.file)
    data_analysis()