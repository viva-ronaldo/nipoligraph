import numpy as np
import feather
import pandas as pd
import re, pickle, json, getpass

from sklearn.decomposition import PCA
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk
if getpass.getuser() == 'david':
    nltk.data.path.append('/media/shared_storage/data/nltk_data/')
else:
    nltk.data.path.append('/home/rstudio/nipol/data/nltk_data/')
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import LdaModel

data_dir = './data/'

mla_ids = pd.read_csv(data_dir + 'all_politicians_list.csv', dtype = {'PersonId': object})
party_group_dict = {
    'Ulster Unionist Party': 'Unionist',
    'Traditional Unionist Voice': 'Unionist',
    'Democratic Unionist Party': 'Unionist',
    'Alliance Party': 'Other',
    'Green Party': 'Other',
    'People Before Profit Alliance': 'Other',
    'Independent': 'Other',
    'Sinn Féin': 'Nationalist',
    'Social Democratic and Labour Party': 'Nationalist'
}
mla_ids['PartyGroup'] = mla_ids.PartyName.apply(lambda p: party_group_dict[p])
#Subset to MLAs for now
mla_ids = mla_ids[mla_ids.role == 'MLA']


#i) Score tweets sentiment  
#TODO just do incrementally?
print('Scoring tweets on sentiment')
#tweets = feather.read_dataframe(data_dir + 'mlas_2019_tweets_apr2019min_to_present.feather')[['status_id','text']]
tweets = pd.concat([
    feather.read_dataframe(data_dir + 'tweets_slim_apr2019min_to_3jun2021.feather')[['status_id','text']],
    feather.read_dataframe(data_dir + 'tweets_slim_4jun2021_to_present.feather')[['status_id','text']]
])

analyzer = SentimentIntensityAnalyzer()
tweets['sentiment_vader_compound'] = tweets['text'].apply(lambda t: analyzer.polarity_scores(t)['compound'])

tweets[['status_id','sentiment_vader_compound']].to_csv(data_dir + 'vader_scored_tweets_apr2019min_to_present.csv', index=False)
print(f"Done: {tweets.shape[0]} tweets")

#ii) Score contribs LDA
print('Scoring plenary contribs with LDA topic model')

def assign_most_likely_topic(list_tuples, topic_nums_to_drop=[]):
    if len(list_tuples) == 0:
        return -999
    else:
        results = sorted(list_tuples, key=lambda t: t[1], reverse=True)
        results = [el for el in results if el[0] not in topic_nums_to_drop]
        if len(results) == 0:
            return -999
        else:
            return results[0][0]

#Load model
with open(data_dir + 'contribs_lda_model.pkl','rb') as f:
    lda_stuff = pickle.load(f)

contribs = feather.read_dataframe(data_dir + 'plenary_hansard_contribs.feather')

#First catch special uses of House and Chamber before casing
contribs['text_proc'] = contribs.contrib
contribs['text_proc'] = contribs.text_proc.str.replace('the House','_house_token_')  #to distinguish from housing
contribs['text_proc'] = contribs.text_proc.str.replace('the Chamber','_house_token_') 
contribs['text_proc'] = contribs.text_proc.str.replace('this House','_house_token_') 
contribs['text_proc'] = contribs.text_proc.str.replace('this Chamber','_house_token_') 
#Lower case and remove br tags
contribs['text_proc'] = contribs.text_proc.str.lower()
contribs['text_proc'] = contribs.text_proc.str.replace('\r<br />','')
#Remove short lines asking to give way
contribs = contribs[(~contribs.text_proc.str.contains('give way')) |
                    (contribs.text_proc.str.len() > 30)]
#and simple 'i beg to move's; some > 2000 in length contain the argument as well
contribs = contribs[(~contribs.text_proc.str.contains('i beg to move')) |
                    (contribs.text_proc.str.len() > 2000)]
#and anything else very short
contribs = contribs[contribs.text_proc.str.len() >= 10]
#Some manual substitutions before tokenizing
#contribs['text_proc'] = contribs.text_proc.str.replace('covid-19','covid_19')
#contribs['text_proc'] = contribs.text_proc.str.replace('power-sharing','power_sharing')
contribs['text_proc'] = contribs.text_proc.str.replace('go raibh','go_raibh')
contribs['text_proc'] = contribs.text_proc.str.replace('-','_')
contribs['text_proc'] = contribs.text_proc.apply(lambda t: re.sub('£[\.\,\dm]*', '_price_token_', t))
contribs['text_proc'] = contribs.text_proc.apply(lambda t: re.sub('20[\d\-]{2,7}', '_year_token_', t))
#Remove 'leave out all after XXX and insert'
contribs['text_proc'] = contribs.text_proc.apply(lambda t: re.sub('leave out all after .* and insert:?\n\n\"', ' ', t))

#Tokenize documents
stopWords = set(stopwords.words('english'))
#also remove common procedural words
stopWords.update(['thank','mr','mrs','speaker','assembly','welcome','today','motion','give','way','giving',
                  'member','members','bill','amendment','debate','order','raised','committee',
                  'party','sinn','féin','dup',
                  'minister','ministers','deputy','first','chamber','department','departments','statement',
                  'executive','office','question','tabled','table',
                  'cheann','comhairle','raibh','maith','agat','agus','gabhaim','buíochas',
                  'leis','aire','labhraím','fáilte','leascheann','míle','go_raibh',
                  'us','would','time','one','think','said','want','say','know','get','see','need','sure','take','number'
                 ])
tokenizer = WordPunctTokenizer()
contribs['tokenized_text'] = contribs.text_proc.apply(lambda t: [w for w in tokenizer.tokenize(t) if w.isalpha() and w not in stopWords])

#encode using dictionary
corpus = [lda_stuff['dictionary'].doc2bow(doc) for doc in contribs.tokenized_text.tolist()]

#assign topics
contribs['topic_num'] = [assign_most_likely_topic(l, topic_nums_to_drop=lda_stuff['topic_nums_to_drop']) \
                         for l in lda_stuff['topic_model'].get_document_topics(corpus, minimum_probability=0.1)]
contribs['topic_name'] = contribs.topic_num.apply(lambda n: lda_stuff['topic_name_dict'][n])

#save
contribs[['speaker','session_id','topic_name']].to_csv(data_dir + 'lda_scored_plenary_contribs.csv', index=False)
print(f"Done: {contribs.shape[0]} contributions")

#iii) Convert tweets to word vectors then score with PCA
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

#iv) pre-compute v_comms (and hist_v_comms for now)
def analyse_votes(votes_df, mla_ids):
    v_comms = []
    for v_id in votes_df.EventId.unique():
        tmp = votes_df.loc[votes_df.EventId==v_id]
        
        #Tabler(s) are not known for EventId 1084 
        if tmp.tabler_personIDs.iloc[0] == '':
            continue
        
        vote_date = str(tmp.DivisionDate.iloc[0])
        vote_subject = tmp.motion_plus_url.iloc[0]
        vote_result = 'PASS' if tmp.Outcome.iloc[0] in ['The Amendment Was Therefore Agreed',
            'The Motion Was Carried','The Motion Was Carried By Cross Community Consent'] else 'FAIL'
        vote_tabler_group = tmp.tabler_personIDs.iloc[0].split(';')
        vote_tabler_group = [mla_ids.loc[mla_ids.PersonId==x,'PartyGroup'].iloc[0] for x in vote_tabler_group]
        vote_tabler_group = vote_tabler_group[0] if len(np.unique(vote_tabler_group))==1 else 'Mixed'

        alli_vote_count = tmp[tmp.PartyName=='Alliance'].Vote.value_counts()
        green_vote_count = tmp[tmp.PartyName=='Green'].Vote.value_counts()
        uni_vote_count = tmp[tmp.Designation=='Unionist'].Vote.value_counts()
        if uni_vote_count.max()/uni_vote_count.sum() >= 0.95:
            u_b_v = uni_vote_count.sort_values().index[-1]
        else:
            u_b_v = 'ABSTAINED' if uni_vote_count.shape[0] == 0 else 'split'
        nat_vote_count = tmp[tmp.Designation=='Nationalist'].Vote.value_counts()
        if nat_vote_count.max()/nat_vote_count.sum() >= 0.95:
            n_b_v = nat_vote_count.sort_values().index[-1]
        else:
            n_b_v = 'ABSTAINED' if nat_vote_count.shape[0] == 0 else 'split'
        if (alli_vote_count.max() / alli_vote_count.sum() >= 0.9):
            alli_vote = alli_vote_count.sort_values().index[-1]
        else:
            alli_vote = 'ABSTAINED' if alli_vote_count.shape[0] == 0 else 'split'
        if (green_vote_count.max() / green_vote_count.sum() >= 0.9):
            green_vote = green_vote_count.sort_values().index[-1]
        else:
            green_vote = 'ABSTAINED' if green_vote_count.shape[0] == 0 else 'split'
        #
        dupsf_vote_count = tmp[tmp.PartyName.isin(['DUP','Sinn Fein'])].Vote.value_counts()
        if dupsf_vote_count.max()/dupsf_vote_count.sum() >= 0.95:
            pass
            #print('DUP+SF bloc vote', dupsf_vote_count.index[dupsf_vote_count.argmax()])
        v_comms.append((v_id, vote_date, vote_subject, vote_tabler_group, vote_result, u_b_v, n_b_v, alli_vote, green_vote))
    
    v_comms = pd.DataFrame(v_comms, columns=['EventId','vote_date','vote_subject','vote_tabler_group','vote_result','uni_bloc_vote','nat_bloc_vote', 'alli_vote','green_vote'])
    v_comms['uni_nat_split'] = (v_comms.uni_bloc_vote!='split') & \
        (v_comms.nat_bloc_vote!='split') & (v_comms.nat_bloc_vote!=v_comms.uni_bloc_vote)
    v_comms['uni_nat_split'] = v_comms.uni_nat_split.apply(lambda b: 'Yes' if b else 'No')

    v_comms = v_comms.sort_values('vote_date', ascending=False)

    return v_comms

with open(data_dir + 'party_names_translation_short.json', 'r') as f:
    party_names_translation = json.load(f)

votes_df = feather.read_dataframe(data_dir + 'division_votes.feather')
vote_results_df = feather.read_dataframe(data_dir + 'division_vote_results.feather')
vote_results_df = vote_results_df.merge(mla_ids[['PersonId','PartyName']], 
    on='PersonId', how='left')
vote_results_df = vote_results_df[vote_results_df.PartyName.notnull()]  #drop a few with missing member and party names
vote_results_df['PartyName'] = vote_results_df.PartyName.apply(lambda p: party_names_translation[p])
votes_df = votes_df.merge(vote_results_df, on='EventId', how='inner')
votes_df = votes_df.merge(mla_ids[['PersonId','normal_name']], on='PersonId', how='inner')
votes_df['DivisionDate'] = pd.to_datetime(votes_df['DivisionDate'], utc=True)
votes_df = votes_df.sort_values('DivisionDate')
#now simplify to print nicer
votes_df['DivisionDate'] = votes_df['DivisionDate'].dt.date
#To pass all votes list, create a column with motion title and url 
#  joined by | so that I can split on this inside the datatable
votes_df['motion_plus_url'] = votes_df.apply(
    lambda row: f"{row['Title']}|http://aims.niassembly.gov.uk/plenary/details.aspx?&ses=0&doc={row['DocumentID']}&pn=0&sid=vd", axis=1)

print(votes_df.EventId.nunique())
v_comms = analyse_votes(votes_df, mla_ids)
print(v_comms.EventId.nunique())

feather.write_dataframe(v_comms, data_dir + 'division_votes_v_comms.feather')


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
