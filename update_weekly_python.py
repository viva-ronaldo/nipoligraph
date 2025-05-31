# TODO Convert for two historical v_comms files to csv

# Change division_votes_v_comms from feather to csv 
# DONE here
# DONE in app.py
# DONE in weekly sh x2

# More weekly updates
# SKIPPED NOW i) Score tweets sentiment
# SKIPPED NOW ii) Convert tweets to word vectors and score with PCA - TURNED OFF
# iii) Score contributions LDA
# iv) Division votes bloc information (v_comms)

import feather
import pandas as pd
import json, os, pickle, re, yaml
import boto3  # if data_dir is S3

#Only needed if do_twitter
#import numpy as np
#from sklearn.decomposition import PCA
#from nltk.sentiment.vader import SentimentIntensityAnalyzer

#nltk data for stopwords: simplified this by saving the data to S3
#import nltk
#import getpass
#if getpass.getuser() == 'david':
#    nltk.data.path.append('/media/shared_storage/data/nltk_data/')
#else:
#    nltk.data.path.append('/home/rstudio/nipol/data/nltk_data/')

# import functions as a module to get the constants and its package imports
import functions_for_update_weekly_python as weekly_update_fns

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

do_twitter = False

data_dir = f"s3://{config['NIPOL_DATA_BUCKET']}/"

# Tweets files
if do_twitter:
    hist_tweets_slim_filepath = os.path.join(data_dir, 'tweets_slim_apr2019min_to_3jun2021.feather')
    current_tweets_slim_filepath = os.path.join(data_dir, 'tweets_slim_4jun2021_to_present.feather')
    vader_scored_tweets_filepath = os.path.join(data_dir, 'vader_scored_tweets_apr2019min_to_present.csv')

# Plenary contribs files
contribs_filepath = os.path.join(data_dir, 'plenary_hansard_contribs.feather')
lda_scored_contribs_filepath = os.path.join(data_dir, 'lda_scored_plenary_contribs.csv')
contribs_lda_model_filepath = os.path.join(data_dir, 'contribs_lda_model.pkl')
stopwords_filepath = os.path.join(data_dir, 'nltk_english_stopwords.txt')

# Votes (v_comms) files
votes_filepath = os.path.join(data_dir, 'division_votes.feather')
vote_results_filepath = os.path.join(data_dir, 'division_vote_results.feather')
mla_ids_filepath = os.path.join(data_dir, 'all_politicians_list.csv')
party_names_translation_filepath = os.path.join(data_dir, 'party_names_translation_short.json')
v_comms_out_filepath = os.path.join(data_dir, 'division_votes_v_comms.csv')


#i) Score tweets sentiment: read slim tweets files, apply Vader, save as vader_scored file
if do_twitter:
    print('\nScoring tweets on sentiment')
    tweets = pd.concat([
        feather.read_dataframe(hist_tweets_slim_filepath)[['status_id', 'text']],
        feather.read_dataframe(current_tweets_slim_filepath)[['status_id', 'text']]
    ])
    tweets['status_id'] = tweets.status_id.astype(int)
    
    # Old method - score all 500k each time; may take minutes
    #analyzer = SentimentIntensityAnalyzer()
    #tweets['sentiment_vader_compound'] = tweets['text'].apply(lambda t: analyzer.polarity_scores(t)['compound'])
    #tweets[['status_id','sentiment_vader_compound']].to_csv(data_dir + 'vader_scored_tweets_apr2019min_to_present.csv', index=False)
    
    # New method - score new ones only and append
    weekly_update_fns.append_sentiment_scored_tweets(tweets, vader_scored_tweets_filepath)


#ii) Convert tweets to word vectors then score with PCA
if do_twitter:
    pass

    #** 15 May 2021 - turned off temporarily because there isn't enough memory to
    #   run the .from_dict line for all rows at once, and the plot isn't that useful.
    # print('Converting tweets to word vectors and scoring PCA')
    
    # tweets['text_proc'] = tweets.text.str.lower()
    # tweets['text_proc'] = tweets.text_proc.str.replace('#','')
    # tweets['text_proc'] = tweets.text_proc.str.replace('@\w*','')
    # tweets['text_proc'] = tweets.text_proc.str.replace('//t.*','')
    # tweets['text_proc'] = tweets.text_proc.str.replace('\d{4}','_year_token_')
    # wp_tokenizer = WordPunctTokenizer()
    # tweets['text_tokenized'] = tweets.text_proc.apply(wp_tokenizer.tokenize)
    # stopWords = set(stopwords.words('english')).union(set(['-','\'',',','.','!','!!','?','...','"']))
    # tweets['text_tokenized'] = tweets.text_tokenized.apply(lambda l: [w for w in l if w not in stopWords])
    
    # with open(data_dir + 'tweets_wv_pca.pkl','rb') as f:
    #     tweets_pca_stuff = pickle.load(f)
    # tweets_wordvecs = tweets_pca_stuff['tweets_wordvecs']
    # tweets_pca = tweets_pca_stuff['tweets_pca']
    
    # def get_tweet_av_word_vect(token_list, wv):
    #     output_av = np.zeros(wv.vector_size)
    #     count = 0
    #     for t in token_list:
    #         if t in wv.vocab:
    #             output_av += wv[t]
    #             count += 1
    #     return output_av / max(count,1)
    
    # tweets_vectorised = tweets.text_tokenized.apply(lambda tl: get_tweet_av_word_vect(tl, tweets_wordvecs.wv))
    # print(tweets.shape)
    # print(tweets_vectorised.shape)
    # tweets_vectorised = pd.DataFrame.from_dict(dict(zip(tweets_vectorised.index, tweets_vectorised.values)), orient='index')
    
    # tmp = tweets_pca.transform(tweets_vectorised)
    # tweets['wv_PC1'] = tmp[:,0]
    # tweets['wv_PC2'] = tmp[:,1]
    # del tmp
    
    # tweets[['status_id','wv_PC1','wv_PC2']].to_csv(data_dir + 'wv_pca_scored_tweets_apr2019min_to_present.csv', index=False)

#iii) Score contribs LDA
if 's3' in contribs_lda_model_filepath:
    s3 = boto3.client('s3')
    bucket_name = contribs_lda_model_filepath.replace('s3://', '').split('/')[0]
    response = s3.get_object(Bucket=bucket_name, Key=contribs_lda_model_filepath.split('/')[-1])
    lda_model = pickle.loads(response['Body'].read())
else:
    with open(contribs_lda_model_filepath, 'rb') as f:
        lda_model = pickle.load(f)

weekly_update_fns.score_contribs_with_lda(contribs_filepath,
                                          lda_scored_contribs_filepath,
                                          lda_model,
                                          stopwords_filepath)

#iv) pre-compute v_comms (and hist_v_comms for now)

# Prepare v_comms, the group voting table by EventId (vote), and write to file.
# A group with no votes seen is marked as ABSTAINED but this could instead be because they have no members,
#   so need to elsewhere turn off Green results in 2023. Can't tell from votes_df which parties have active MLAs.

weekly_update_fns.analyse_votes_by_bloc(votes_filepath,
                                        vote_results_filepath,
                                        mla_ids_filepath,
                                        party_names_translation_filepath,
                                        v_comms_out_filepath)


# def assign_session_name(date_string):
#     if date_string < '2011-05-05':
#         session_name = '2007-2011'
#     elif date_string < '2016-05-05':
#         session_name = '2011-2016'
#     elif date_string < '2017-03-02':
#         session_name = '2016-2017'
#     elif date_string < '2020-01-11':
#         session_name = '2017-2019'
#     else:
#         session_name = '2020-2022'
#     return session_name

# hist_mla_ids = feather.read_dataframe(data_dir + 'hist_mla_ids_by_session.feather')
# hist_mla_ids['PartyGroup'] = hist_mla_ids.PartyName.apply(lambda p: party_group_dict[p])

# hist_votes_df = feather.read_dataframe(data_dir + 'historical_division_votes.feather')
# hist_votes_df = hist_votes_df.rename(index=str, columns={'EventID':'EventId'})
# hist_vote_results_df = feather.read_dataframe(data_dir + 'historical_division_vote_results.feather')
# #Reorder operations from above
# hist_votes_df = hist_votes_df.merge(hist_vote_results_df, on='EventId', how='inner')

# hist_votes_df['session_name'] = hist_votes_df.DivisionDate.apply(assign_session_name)

# hist_votes_df = hist_votes_df.merge(hist_mla_ids[['PersonId','PartyName','normal_name','session_name']], 
#     on=['PersonId','session_name'], how='left')
# hist_votes_df = hist_votes_df[hist_votes_df.PartyName.notnull()]
# hist_votes_df['PartyName'] = hist_votes_df.PartyName.apply(lambda p: party_names_translation[p])

# hist_votes_df['DivisionDate'] = pd.to_datetime(hist_votes_df['DivisionDate'], utc=True)
# hist_votes_df = hist_votes_df.sort_values('DivisionDate')
# #now simplify to print nicer
# hist_votes_df['DivisionDate'] = hist_votes_df['DivisionDate'].dt.date
# #To pass all votes list, create a column with motion title and url 
# #  joined by | so that I can split on this inside the datatable
# hist_votes_df['motion_plus_url'] = hist_votes_df.apply(
#     lambda row: f"{row['Title']}|http://aims.niassembly.gov.uk/plenary/details.aspx?&ses=0&doc={row['DocumentID']}&pn=0&sid=vd", axis=1)

# feather.write_dataframe(hist_v_comms, data_dir + 'historical_division_votes_v_comms.feather')
