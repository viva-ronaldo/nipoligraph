# TODO update news page and indiv news tracker to start from 2024
#   and remove under development banner
# Then do the news summaries

from flask import Flask, render_template, url_for, \
    request, flash, jsonify, redirect, send_from_directory
import requests
import datetime, re, getpass, os, time
import feather, json, pickle, yaml
import pandas as pd
import altair as alt
import numpy as np
from itertools import product
from collections import defaultdict
from sklearn.decomposition import PCA

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

app = Flask(__name__)
app.debug = False
app.secret_key = 'dont_need_one'
app.url_map.strict_slashes = False

#Load data
#---------
#Files used:
#Rarely updated:
#- all_politicians_list.csv (names and ids)
#- party_group_dict.json (uni/nat labels)
#- party_names_translation_short.json
#- party_names_translation.json
#- mp_api_numbers.json
#- member_other_photo_links.json
#- party_colours.csv
#- politicians_twitter_accounts_ongoing.csv
#- contribs_lda_model.pkl
#- newsriver_articles_[may,june,july]2020.feather
#- election_details.csv
#- election_results.csv
#- poll_details.csv
#- poll_results.csv
#- hist* things
#Regular update:
#- mlas_2019_tweets_apr2019min_to_present.feather
#- vader_scored_tweets_apr2019min_to_present.csv
#- niassembly_questions.feather
#- niassembly_answers.feather
#- division_votes.feather
#- division_vote_results.feather
#- division_votes_v_comms.csv
#- lda_scored_plenary_contribs.csv
#- plenary_hansard_contribs_emotions_averaged.feather
#- diary_future_events.psv
#- current_ministers_and_speakers.csv
#- current_committee_memberships.csv
#- newsriver_articles_ongoing2020.feather
#- tweets_network_last3months_top5s.csv
#- static/tweets_network_last3months_edges.csv
#- static/tweets_network_last3months_nodes.json

print('Starting...')
if getpass.getuser() == 'david':
    data_dir = '/media/shared_storage/data/nipol_aws_copy/data/'
    test_mode = True
else:
    data_dir = '/home/vivaronaldo/nipol/data/'
    test_mode = False

pca_votes_threshold_fraction = 0.70
news_volume_average_window_weeks = 7
poll_track_timestep = 100 if test_mode else 10
#Assembly page sessions
valid_session_names = config['ALL_ASSEMBLY_SESSIONS'] #order for dropdown menu
CURRENT_ASSEMBLY_SESSION = config['CURRENT_ASSEMBLY_SESSION']

#Election forecast is usually off
show_election_forecast = config['INCLUDE_ELECTION_FORECAST']
# From mid-2023, can't scrape tweets, so exclude from indiv page and note on twitter page
include_twitter = config['INCLUDE_TWITTER']
# TODO do more with this

#use the colours we get from 'tableau20' but make repeatable and for reuse on indiv page
plenary_contribs_colour_dict = {
    'finance': '#88d27a',
    'justice & legislation': '#e45756',
    'infrastructure & investment': '#83bcb6',
    'government business': '#b79a20',
    'health & social care': '#f2cf5b',
    'education': '#54a24b',
    'politics/parties': '#ff9d98',
    'public sector & unions': '#79706e',
    'brexit/trade': '#f58518',
    'economy': '#ffbf79',
    'agriculture, prisons, industry': '#4c78a8',
    'belfast city': '#9ecae9',
    'housing': '#439894'
}

diary_colour_dict = defaultdict(lambda: 'Black')
diary_colour_dict.update(
    {'Committee for Agriculture, Environment and Rural Affairs Meeting': 'SeaGreen',
    'Committee for Communities Meeting': 'DeepSkyBlue',
    'Committee for Education Meeting': 'Salmon',
    'Committee for Finance Meeting': 'Purple',
    'Committee for Health Meeting': 'MediumVioletRed',
    'Committee for Infrastructure Meeting': 'Olive',
    'Committee for Justice Meeting': 'MidnightBlue',
    'Committee for the Economy Meeting': 'DarkSlateGrey',
    'Business Committee Meeting':  'Brown',
    'Sitting of the Assembly': 'Black'}
)

mla_ids = pd.read_csv(data_dir + 'all_politicians_list.csv', dtype = {'PersonId': object})
# Fix awkward names in apg and comm attendees which don't use PersonId so can fall out of sync when AIMS changes a name
# This needs to list any new or old names used for an MLA and point to their current PersonId name; include inactive
mla_name_fix_dict = {
    'Lord Elliott of Ballinamallard': mla_ids[mla_ids.PersonId=='128'].iloc[-1].normal_name,
    'Lord Tom Elliott of Ballinamallard': mla_ids[mla_ids.PersonId=='128'].iloc[-1].normal_name,
    'Tom Elliott': mla_ids[mla_ids.PersonId=='128'].iloc[-1].normal_name,
}
#If have too many non-MLA/MPs, could become unreliable over time, so limit to active here
mla_ids = mla_ids[mla_ids.active==1]
mla_ids['fullname'] = mla_ids['MemberFirstName'] + ' ' + mla_ids['MemberLastName']  # slightly different from normal_name

#Add email address
mla_ids = mla_ids.merge(
    pd.read_csv(data_dir + 'mla_email_addresses.csv', dtype = {'PersonId': object}),
    on = 'PersonId', how = 'left'
    )
mla_ids.AssemblyEmail.fillna('none', inplace=True)
mla_ids = mla_ids.merge(
    pd.read_csv(data_dir + 'mp_email_addresses.csv'), on = 'normal_name', how = 'left'
    )
mla_ids.WestminsterEmail.fillna('none', inplace=True)

with open(data_dir + 'party_group_dict.json', 'r') as f:
    party_group_dict = json.load(f)
#Do this before turning PartyName to short form
mla_ids['PartyGroup'] = mla_ids.PartyName.apply(lambda p: party_group_dict[p])

#Handle the two forms of some party names
#with open(data_dir + 'party_names_translation.json', 'r') as f:
with open(data_dir + 'party_names_translation_short.json', 'r') as f:
    party_names_translation = json.load(f)
with open(data_dir + 'party_names_translation.json', 'r') as f:
    party_names_translation_long = json.load(f)
#Numbers to find MP portraits on parliament website
with open(data_dir + 'mp_api_numbers.json', 'r') as f:
    mp_api_numbers = json.load(f)
with open(data_dir + 'member_other_photo_links.json', 'r') as f:
    member_other_photo_links = json.load(f)

mla_ids['PartyName_long'] = mla_ids.PartyName.apply(lambda p: party_names_translation_long[p])
mla_ids['PartyName'] = mla_ids.PartyName.apply(lambda p: party_names_translation[p])

party_colours = pd.read_csv(data_dir + 'party_colours.csv')
col_corrects = {'yellow3':'#e6c300', 'green2':'chartreuse'}
party_colours['colour'] = party_colours['colour'].apply(
    lambda c: c if c not in col_corrects.keys() else col_corrects[c]
)

mla_minister_roles = pd.read_csv(data_dir + 'current_ministers_and_speakers.csv', dtype={'PersonId': object})
mla_minister_roles = mla_minister_roles.merge(mla_ids[['PersonId','normal_name']], on='PersonId', how='inner')
mla_minister_roles = {i[1]['normal_name']: i[1]['AffiliationTitle'] for i in mla_minister_roles.iterrows()}

print('Done ids')

#Twitter
#-------
#tweets_df = feather.read_dataframe(data_dir + 'mlas_2019_tweets_apr2019min_to_present_slim.feather')
tweets_df = pd.concat([
    feather.read_dataframe(data_dir + 'tweets_slim_apr2019min_to_3jun2021.feather'),
    feather.read_dataframe(data_dir + 'tweets_slim_4jun2021_to_present.feather')
])
twitter_ids = pd.read_csv(data_dir + 'politicians_twitter_accounts_ongoing.csv',
    dtype = {'user_id': object})
#tweets_df = tweets_df.merge(twitter_ids[['user_id','mla_party','mla_name']].rename(index=str, columns={'mla_name':'normal_name'}), 
tweets_df = tweets_df.merge(twitter_ids[['user_id','mla_party','normal_name']],
    on='user_id', how='left')
tweets_df['mla_party'] = tweets_df.mla_party.apply(lambda p: party_names_translation[p])
#Filter to 1 July onwards - fair comparison for all
tweets_df = tweets_df[tweets_df.created_ym >= '202007']
tweets_df = tweets_df[tweets_df['normal_name'].isin(mla_ids['normal_name'])]
tweets_df['tweet_type'] = tweets_df.is_retweet.apply(lambda b: 'retweet' if b else 'original')
tweets_df['created_at_week'] = tweets_df['created_at'].dt.isocalendar().week
#tweets_df['created_at_week'] = tweets_df['created_at'].dt.week
#early Jan can be counted as week 52 or 53 by pd.week - messes things up
tweets_df.loc[(tweets_df.created_at_week >= 52) & (tweets_df.created_at.dt.day <= 7), 'created_at_week'] = 1
tweets_df['created_at_yweek'] = tweets_df.apply(
    lambda row: '{:s}-{:02g}'.format(row['created_ym'][:4], row['created_at_week']), axis=1)

last_5_yweeks_tweets = tweets_df.created_at_yweek.sort_values().unique()[-5:]
print('Retweets are counted over',last_5_yweeks_tweets)

tweet_sentiment = pd.read_csv(data_dir + 'vader_scored_tweets_apr2019min_to_present.csv', dtype={'status_id': object})
tweets_df = tweets_df.merge(tweet_sentiment, on='status_id', how='left')

#Which people tweet the most
top_tweeters = tweets_df[tweets_df.created_at_yweek.isin(last_5_yweeks_tweets)].groupby(['normal_name','mla_party','tweet_type']).status_id.count()\
    .reset_index().rename(index=str, columns={'status_id':'n_tweets'})
#(now filter to top 10/15 in function below)
top_tweeters['tooltip_text'] = top_tweeters.apply(
    lambda row: f"{row['normal_name']} ({row['mla_party']}): {row['n_tweets']} {'original tweets' if row['tweet_type'] == 'original' else 'retweets'}", axis=1
)

member_retweets = tweets_df[(~tweets_df.is_retweet) & (tweets_df.created_at_yweek.isin(last_5_yweeks_tweets))]\
    .groupby(['normal_name','mla_party'])\
    .agg({'retweet_count': np.mean, 'status_id': len}).reset_index()\
    .query('status_id >= 10')\
    .rename(index=str, columns={'status_id': 'n_original_tweets', 'retweet_count': 'retweets_per_tweet'})\
    .sort_values('retweets_per_tweet', ascending=False)
member_retweets['tooltip_text'] = member_retweets.apply(
    lambda row: f"{row['normal_name']}: {row['retweets_per_tweet']:.1f} retweets per original tweet (from {row['n_original_tweets']} tweets)", axis=1
)

member_tweet_sentiment = tweets_df[(tweets_df.created_at_yweek.isin(last_5_yweeks_tweets)) & (tweets_df.sentiment_vader_compound.notnull())]\
    .groupby(['normal_name','mla_party'])\
    .agg({'sentiment_vader_compound': np.mean, 'status_id': len}).reset_index()\
    .query('status_id >= 10').sort_values('sentiment_vader_compound', ascending=False)
#Normalise here - only using for one plot (and rankings on indiv pages)
member_tweet_sentiment['sentiment_vader_compound'] = member_tweet_sentiment['sentiment_vader_compound'] - member_tweet_sentiment['sentiment_vader_compound'].mean()
if len(member_tweet_sentiment) > 0:
    member_tweet_sentiment['tooltip_text'] = member_tweet_sentiment.apply(
        lambda row: f"{row['normal_name']}: mean score rel. to avg. = {row['sentiment_vader_compound']:+.2f} ({row['status_id']} tweets)", 
        axis=1
    )
else:
    member_tweet_sentiment['tooltip_text'] = []
# retweet_rate_last_month = tweets_df[(~tweets_df.is_retweet) & (tweets_df.created_at_yweek.isin(last_5_yweeks_tweets))]\
#     .groupby('mla_party')\
#     .agg({'retweet_count': np.mean, 'status_id': len}).reset_index()\
#     .query('status_id >= 10')\
#     .rename(index=str, columns={'status_id': 'n_original_tweets', 'retweet_count': 'retweets_per_tweet'})
# retweet_rate_last_month['tooltip_text'] = retweet_rate_last_month.apply(
#     lambda row: f"{row['mla_party']}: {row['n_original_tweets']} original tweets with average of {row['retweets_per_tweet']:.1f} retweets per tweet",
#     axis=1
# )

#Average tweet PCA position
# tweets_w_wv_pcas = pd.read_csv(data_dir + 'wv_pca_scored_tweets_apr2019min_to_present.csv',
#     dtype={'status_id': object})
# tweet_pca_positions = tweets_df\
#     .merge(tweets_w_wv_pcas, on='status_id')\
#     .groupby(['normal_name','mla_party'])\
#     .agg(
#         mean_PC1 = pd.NamedAgg('wv_PC1', np.mean),
#         mean_PC2 = pd.NamedAgg('wv_PC2', np.mean),
#         num_tweets = pd.NamedAgg('status_id', len)
#     ).reset_index()
# tweet_pca_positions = tweet_pca_positions[tweet_pca_positions.num_tweets >= 20]
# tweet_pca_positions['indiv_page_url'] = ['/individual?mla_name=' + n.replace(' ','+') for n in tweet_pca_positions.normal_name]
# tweet_pca_positions['tooltip_text'] = tweet_pca_positions.apply(
#     lambda row: f"{row['normal_name']} ({row['num_tweets']} tweets since July 2020)", axis=1
# )

tweets_network_top5s = pd.read_csv(data_dir + 'tweets_network_last3months_top5s.csv')

#Generated tweets
gen_tweets = pd.read_csv(data_dir + 'fiveparties_generated_tweets_1epoch_gpt2medium_reppen1pt5.csv')
#exclude any that are all hashtags/mentions
gen_tweets['all_h_or_m'] = gen_tweets.generated_text.apply(lambda t: all([w[0] in ['@','#'] for w in t.split()]))
gen_tweets = gen_tweets[~gen_tweets.all_h_or_m]

print('Done Twitter')

#Assembly 
#--------
questions_df = feather.read_dataframe(data_dir + 'niassembly_questions.feather')

questions_df = questions_df.merge(mla_ids[['PersonId','normal_name','PartyName']],
    left_on = 'TablerPersonId', right_on = 'PersonId', how='left')
#For plot facet labels
questions_df['RequestedAnswerType'] = questions_df.RequestedAnswerType.apply(
    lambda rat: 'Oral' if rat=='oral' else ('Written' if rat=='written' else rat)
)

#NB important to have nunique here to avoid counting a question to FM/DFM twice
questioners = questions_df.groupby(['normal_name','PartyName','RequestedAnswerType']).DocumentId.nunique()\
    .reset_index().rename(index=str, columns={
        'DocumentId': 'Questions asked',
        'RequestedAnswerType': 'Question type'  #for plot titles
    })
#(get top 15 by each question type - now done in function below)
if len(questioners) > 0:
    questioners['tooltip_text'] = questioners.apply(
        lambda row: f"{row['normal_name']}: {row['Questions asked']} {row['Question type'].lower()} question{('s' if row['Questions asked'] != 1 else ''):s} asked", axis=1
    )
else:
    questioners['tooltip_text'] = []

answers_df = feather.read_dataframe(data_dir + 'niassembly_answers.feather')

answers_df = answers_df.merge(mla_ids[['PersonId','normal_name','PartyName']]\
        .rename(index=str, columns={'normal_name': 'Tabler_normal_name', 
            'PersonId': 'TablerPersonId',' PartyName': 'Tabler_party_name'}), 
    on='TablerPersonId', how='inner')
answers_df = answers_df.merge(mla_ids[['PersonId','normal_name','PartyName']]\
        .rename(index=str, columns={'normal_name': 'Minister_normal_name', 
            'PersonId': 'MinisterPersonId', 'PartyName': 'Minister_party_name'}), 
    on='MinisterPersonId', how='left')
#answers_df['Days_to_answer'] = (pd.to_datetime(answers_df['AnsweredOnDate'], utc=True) - pd.to_datetime(answers_df['TabledDate'], utc=True)).dt.days   #fails to convert if there is a mix of offsets
answers_df['Days_to_answer'] = answers_df.apply(lambda row: (pd.to_datetime(row['AnsweredOnDate'][:10]) - pd.to_datetime(row['TabledDate'][:10])).days, axis=1)
#Filtering to current session (there are some 2017-2019 entries but all have missing MinisterPersonId anyway)
#answers_df = answers_df.query("TabledDate > '2022-06-01'")

# minister_answers = answers_df[answers_df.MinisterTitle != 'Assembly Commission']\
#     .groupby(['Minister_normal_name','Minister_party_name']).DocumentId.count().reset_index()\
#     .rename(index=str, columns={'DocumentId':'Questions answered'})
# minister_answers['tooltip_text'] = minister_answers.apply(
#     lambda row: f"{row['Minister_normal_name']}: {row['Questions answered']} answer{('s' if row['Questions answered'] != 1 else ''):s}", axis=1
# )
# minister_answers = minister_answers[minister_answers.Minister_normal_name.isin(mla_minister_roles.keys())]

# minister_time_to_answer = answers_df[answers_df.MinisterTitle != 'Assembly Commission']\
#     .groupby(['Minister_normal_name','Minister_party_name']).Days_to_answer.median().reset_index()\
#     .rename(index=str, columns={'Days_to_answer':'Median days to answer'})
minister_time_to_answer = answers_df[answers_df.MinisterTitle != 'Assembly Commission']\
    .groupby(['Minister_normal_name','Minister_party_name'])\
    .agg(
        median_days_to_answer = pd.NamedAgg('Days_to_answer', np.median),
        num_questions_answered = pd.NamedAgg('Days_to_answer', len)
        ).reset_index()\
    .rename(index=str, columns={
        'median_days_to_answer': 'Median days to answer',  
        'num_questions_answered': 'Questions answered'    #for plot axis title
    })
if len(minister_time_to_answer) > 0:
    minister_time_to_answer['tooltip_text'] = minister_time_to_answer.apply(
        lambda row: f"{row['Minister_normal_name']}: median {row['Median days to answer']:g} day{('s' if row['Median days to answer'] != 1 else ''):s}", axis=1
    )
else:
    minister_time_to_answer['tooltip_text'] = []
minister_time_to_answer = minister_time_to_answer[minister_time_to_answer.Minister_normal_name.isin(mla_minister_roles.keys())]
print('Done questions and answers')

#Votes
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
if len(votes_df) > 0:
    votes_df['motion_plus_url'] = votes_df.apply(
        lambda row: f"{row['Title']}|https://aims.niassembly.gov.uk/plenary/details.aspx?&ses=0&doc={row['DocumentID']}&pn=0&sid=vd", axis=1)
else:
    votes_df['motion_plus_url'] = []

#Votes PCA
votes_df['vote_num'] = votes_df.Vote.apply(lambda v: {'NO':-1, 'AYE':1, 'ABSTAINED':0}[v]) 
votes_pca_df = votes_df[['normal_name','EventId','vote_num']]\
    .pivot(index='normal_name',columns='EventId',values='vote_num').fillna(0) 
tmp = votes_df.normal_name.value_counts()
those_in_threshold_pct_votes = tmp[tmp >= votes_df.EventId.nunique()*pca_votes_threshold_fraction].index
votes_pca_df = votes_pca_df[votes_pca_df.index.isin(those_in_threshold_pct_votes)]

my_pca = PCA(n_components=2, whiten=True)  #doesn't change results but axis units closer to 1
my_pca.fit(votes_pca_df)
#print(my_pca.explained_variance_ratio_)
mlas_2d_rep = pd.DataFrame({'x': [el[0] for el in my_pca.transform(votes_pca_df)],
                            'y': [el[1] for el in my_pca.transform(votes_pca_df)],
                            'normal_name': votes_pca_df.index,
                            'indiv_page_url': ['/individual?mla_name=' + n.replace(' ','+') for n in votes_pca_df.index],
                            'party': [mla_ids.loc[mla_ids.normal_name == n].PartyName.iloc[0] for n in votes_pca_df.index]})

#Votes party unity 
# votes_party_unity = votes_df.groupby(['PartyName','EventId'])\
#     .agg({'Vote': lambda v: len(np.unique(v)),'normal_name': len}).reset_index()\
#     .query('normal_name >= 5')\
#     .groupby('PartyName').agg({'Vote': lambda v: 100*np.mean(v==1), 'EventId': len}).reset_index()
# votes_party_unity.columns = ['PartyName', 'Percent voting as one', 'n_votes']
# votes_party_unity['tooltip_text'] = votes_party_unity.apply(
#     lambda row: f"{row['PartyName']} voted as one in {row['Percent voting as one']:.0f}% of {row['n_votes']} votes",
#     axis=1
# )

#Vote commentary - pre-computed
#v_comms = feather.read_dataframe(data_dir + 'division_votes_v_comms.feather')
v_comms = pd.read_csv(data_dir + 'division_votes_v_comms.csv')

#Contributions - plenary sessions - don't need the text
#plenary_contribs_df = feather.read_dataframe(data_dir + 'plenary_hansard_contribs_201920sessions_topresent.feather')
#plenary_contribs_df = plenary_contribs_df[plenary_contribs_df.speaker.isin(mla_ids.normal_name)]

scored_plenary_contribs_df = pd.read_csv(data_dir + 'lda_scored_plenary_contribs.csv')
#scored_plenary_contribs_df = scored_plenary_contribs_df[scored_plenary_contribs_df.speaker.isin(mla_ids.normal_name)]

plenary_contribs_topic_counts =  scored_plenary_contribs_df.groupby('topic_name').count().reset_index()\
    [['topic_name','session_id']].rename(index=str, columns={'session_id': 'n_contribs'})
#Load model to get the topic top words
with open(data_dir + 'contribs_lda_model.pkl','rb') as f:
    lda_stuff = pickle.load(f)
lda_top5s = [(el[0], ', '.join([f"'{t[0]}'" for t in el[1]])) for el in lda_stuff['topic_model'].show_topics(num_topics=lda_stuff['topic_model'].num_topics, num_words=5, formatted=False)] 
lda_top5s = [(lda_stuff['topic_name_dict'][el[0]], el[1]) for el in lda_top5s]
plenary_contribs_topic_counts = plenary_contribs_topic_counts.merge(
    pd.DataFrame(lda_top5s, columns=['topic_name','top5_words'])
)
if len(plenary_contribs_topic_counts) > 0:
    plenary_contribs_topic_counts['tooltip_text'] = plenary_contribs_topic_counts.apply(
        lambda row: f"{row['topic_name']}: strongest words are {row['top5_words']}", axis=1
    )
else:
    plenary_contribs_topic_counts['tooltip_text'] = []

#plenary contrib emotions
emotions_df = feather.read_dataframe(data_dir + 'plenary_hansard_contribs_emotions_averaged.feather')
#TODO empty file? It should get remade next update when there are some entries

emotions_df = emotions_df.merge(mla_ids[['normal_name','PartyName']], 
    left_on='speaker', right_on='normal_name', how='inner')
emotions_party_agg = emotions_df.groupby(['PartyName','emotion_type']).apply(
    lambda x_grouped: np.average(x_grouped['ave_emotion'], weights=x_grouped['word_count']))\
    .reset_index()\
    .rename(index=str, columns={0:'ave_emotion'})
print('Done votes and contributions')

if os.stat(data_dir + 'diary_future_events.psv').st_size > 10:
    diary_df = pd.read_csv(data_dir + 'diary_future_events.psv', sep='|')
else:
    diary_df = pd.DataFrame(columns=['EventId','EventDate','EventType',
        'EventTypeId','AddressId','LocationRoom','Address1',
        'TownCity','OrganisationName','StartTime','OrganisationId',
        'EndTime','Weekday','Address'])
#Use StartTime to get the date to avoid problems with midnight +/-1h BST
diary_df['EventPrettyDate'] = pd.to_datetime(diary_df['StartTime'], utc=True).dt.strftime('%A, %-d %B')
if len(diary_df) > 0:
    diary_df['EventName'] = diary_df.apply(
        lambda row: row['OrganisationName']+' Meeting' if row['EventType']=='Committee Meeting' else row['EventType'], 
        axis=1
    )
else:
    diary_df['EventName'] = []
diary_df['EventHTMLColour'] = diary_df.EventName.apply(lambda e: diary_colour_dict[e])
#Exclude events that have now happened (will run filter again in assembly.html function)
#diary_df = diary_df[diary_df['EventDate'] >= datetime.date.today().strftime('%Y-%m-%d')]
#diary_df = diary_df[diary_df['EndTime'] >= datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')]

committee_roles = pd.read_csv(data_dir + 'current_committee_memberships.csv', dtype={'PersonId': object})
committee_roles = committee_roles.merge(mla_ids[['PersonId','normal_name']], on='PersonId', how='inner')

committee_meeting_attendance = pd.read_csv(data_dir + 'committee_meetings_attendances_feb2024topresent.csv')
committee_meeting_attendance['member'] = committee_meeting_attendance.member.str.replace('Mr |Mrs |Ms |Miss |Dr |Sir | OBE| MBE| CBE| MC', '', regex=True)
committee_meeting_attendance = (committee_meeting_attendance
    .assign(member = lambda df: df.member.apply(lambda m: mla_name_fix_dict.get(m, m)))
    .rename(columns={'member': 'fullname'})
    .merge(mla_ids[['fullname', 'normal_name', 'MemberLastName']], on='fullname', how='inner')
    .drop(columns='fullname')
    .assign(meeting_date = lambda df: pd.to_datetime(df.meeting_date, format='%d/%m/%Y'))
    )
committee_attendees_agg = (committee_meeting_attendance
    .query('attended')
    .groupby('meeting_id')
    .agg(attendees = ('MemberLastName', lambda m: ', '.join(m)))
    .assign(attendees = lambda df: df.attendees.fillna('-'))
    )

committee_meeting_agendas = pd.read_csv(data_dir + 'committee_meetings_agendas_feb2024topresent.csv')
def truncate_agenda_list(agenda_items):
    agenda_items = agenda_items.tolist()
    if len(agenda_items) > 5:
        agenda_items = agenda_items[:4] + [f'<i>and {len(agenda_items)-4} more items</i>']
    return '<ul><li>' + '</li><li>'.join(agenda_items) + '</li></ul>'
committee_agendas_agg = (committee_meeting_agendas
    .assign(meeting_date = lambda df: pd.to_datetime(df.meeting_date, format='%d/%m/%Y'))
    .groupby(['meeting_id', 'committee_name', 'meeting_date'], as_index=False)
    .agg(agenda_list = ('agenda_item', truncate_agenda_list))
    .merge(committee_attendees_agg, on='meeting_id', how='left')
    .sort_values(['meeting_date', 'meeting_id'], ascending=False)
    .assign(meeting_date_and_url = lambda df: df.apply(lambda row: f"{row.meeting_date.strftime('%d %B %Y')}|https://aims.niassembly.gov.uk/committees/meetings.aspx?cid=0&mid={row.meeting_id}", axis=1))
    .filter(['meeting_date_and_url', 'committee_name', 'agenda_list', 'attendees'])
    )

mla_interests = pd.read_csv(data_dir + 'current_mla_registered_interests.csv', dtype={'PersonId': object})
mla_interests = mla_interests.merge(mla_ids[['PersonId','normal_name']], on='PersonId', how='inner')

allpartygroup_memberships = (
    pd.read_csv(data_dir + 'current_apg_group_memberships.csv')
    .assign(member = lambda df: df.member.apply(lambda m: mla_name_fix_dict.get(m, m)))
    .rename(columns={'member': 'fullname'})
    .merge(mla_ids[['fullname', 'normal_name']], on='fullname', how='inner')
    .drop(columns='fullname')
    .assign(apg_name = lambda df: df.apg_name.str.replace('All-Party [gG]roup on ', '', regex=True))
    )

#Assembly historical
#-------------------

#Read in original hist files for 2007-2019 sessions, and append
#  later sessions that become historical, which is 2020-2022

#Session dates 
def assign_session_name(date_string):
    if date_string < '2007-03-07':
        session_name = 'pre-2007'
    elif date_string < '2011-05-05':
        session_name = '2007-2011'
    elif date_string < '2016-05-05':
        session_name = '2011-2016'
    elif date_string < '2017-03-02':
        session_name = '2016-2017'
    elif date_string < '2020-01-11':
        session_name = '2017-2019'
    elif date_string < '2022-06-01':
        session_name = '2020-2022'
    else:
        session_name = '2022-2027'
    return session_name
#There were some questions submitted in the 2017-2019 period but don't plot this session

hist_mla_ids = pd.concat([
    feather.read_dataframe(data_dir + 'hist_mla_ids_by_session.feather'),
    feather.read_dataframe(data_dir + 'hist_20202022_mla_ids_by_session.feather')
    ])
print(f'{len(hist_mla_ids)} rows in hist mla ids')

hist_mla_ids['PartyGroup'] = hist_mla_ids.PartyName.apply(lambda p: party_group_dict[p])
hist_mla_ids['PartyName_long'] = hist_mla_ids.PartyName.apply(lambda p: party_names_translation_long[p])
hist_mla_ids['PartyName'] = hist_mla_ids.PartyName.apply(lambda p: party_names_translation[p])

hist_questions_df = pd.concat([
    feather.read_dataframe(data_dir + 'historical_niassembly_questions_asked.feather'),
    feather.read_dataframe(data_dir + 'historical_20202022_niassembly_questions_asked.feather')
    ])
print(f'{len(hist_questions_df)} rows in hist questions')
hist_questions_df['session_name'] = hist_questions_df.TabledDate.apply(assign_session_name)
#Join after adding session_name
hist_questions_df = hist_questions_df.merge(hist_mla_ids[['PersonId','normal_name','PartyName','session_name']],
    left_on = ['TablerPersonId','session_name'],
    right_on = ['PersonId','session_name'], how='left')
#For plot facet labels
hist_questions_df['RequestedAnswerType'] = hist_questions_df.RequestedAnswerType.apply(
    lambda rat: 'Oral' if rat=='oral' else ('Written' if rat=='written' else rat)
)
#Historical questions by member
hist_questioners = hist_questions_df.groupby(['session_name','normal_name','PartyName','RequestedAnswerType']).DocumentId.nunique()\
    .reset_index().rename(index=str, columns={
        'DocumentId': 'Questions asked',
        'RequestedAnswerType': 'Question type'  #for plot titles
    })
hist_questioners['tooltip_text'] = hist_questioners.apply(
    lambda row: f"{row['normal_name']}: {row['Questions asked']} {row['Question type'].lower()} question{('s' if row['Questions asked'] != 1 else ''):s} asked", axis=1
)

#Historical answers
hist_answers_df = pd.concat([
    feather.read_dataframe(data_dir + 'historical_niassembly_answers.feather'),
    feather.read_dataframe(data_dir + 'historical_20202022_niassembly_answers.feather')
    ])
print(f'{len(hist_answers_df)} rows in hist answers')
hist_answers_df['session_name'] = hist_answers_df.AnsweredOnDate.apply(assign_session_name)
hist_answers_df = hist_answers_df.merge(hist_mla_ids[['PersonId','normal_name','PartyName','session_name']]\
        .rename(index=str, columns={'normal_name': 'Tabler_normal_name', 
            'PersonId': 'TablerPersonId',' PartyName': 'Tabler_party_name'}), 
    on=['TablerPersonId','session_name'], how='inner')
hist_answers_df = hist_answers_df.merge(hist_mla_ids[['PersonId','normal_name','PartyName','session_name']]\
        .rename(index=str, columns={'normal_name': 'Minister_normal_name', 
            'PersonId': 'MinisterPersonId', 'PartyName': 'Minister_party_name'}), 
    on=['MinisterPersonId','session_name'], how='left')
hist_answers_df['Days_to_answer'] = (pd.to_datetime(hist_answers_df['AnsweredOnDate']) - pd.to_datetime(hist_answers_df['TabledDate'])).dt.days 

hist_minister_time_to_answer = hist_answers_df[hist_answers_df.MinisterTitle != 'Assembly Commission']\
    .groupby(['session_name','Minister_normal_name','Minister_party_name'])\
    .agg(
        median_days_to_answer = pd.NamedAgg('Days_to_answer', np.median),
        num_questions_answered = pd.NamedAgg('Days_to_answer', len)
        ).reset_index()\
    .rename(index=str, columns={
        'median_days_to_answer': 'Median days to answer',  
        'num_questions_answered': 'Questions answered'    #for plot axis title
    })
hist_minister_time_to_answer['tooltip_text'] = hist_minister_time_to_answer.apply(
    lambda row: f"{row['Minister_normal_name']}: median {row['Median days to answer']:g} day{('s' if row['Median days to answer'] != 1 else ''):s}", axis=1
)

#print(hist_minister_time_to_answer[hist_minister_time_to_answer.session_name=='2011-2016'])
print('Done historical questions and answers')

#Historical votes
hist_votes_df = pd.concat([
    feather.read_dataframe(data_dir + 'historical_division_votes.feather'),
    feather.read_dataframe(data_dir + 'historical_20202022_division_votes.feather')
    ])
print(f'{len(hist_votes_df)} rows in hist votes')
hist_votes_df = hist_votes_df.rename(index=str, columns={'EventID':'EventId'})

hist_vote_results_df = pd.concat([
    feather.read_dataframe(data_dir + 'historical_division_vote_results.feather'),
    feather.read_dataframe(data_dir + 'historical_20202022_division_vote_results.feather')
    ])
print(f'{len(hist_vote_results_df)} rows in hist vote_results')
#Reorder operations from above
hist_votes_df = hist_votes_df.merge(hist_vote_results_df, on='EventId', how='inner')

hist_votes_df['session_name'] = hist_votes_df.DivisionDate.apply(assign_session_name)
hist_votes_df = hist_votes_df[hist_votes_df.session_name.isin(valid_session_names)].copy()
#don't use pre-2007

hist_votes_df = hist_votes_df.merge(hist_mla_ids[['PersonId','PartyName','normal_name','session_name']], 
    on=['PersonId','session_name'], how='left')
hist_votes_df = hist_votes_df[hist_votes_df.PartyName.notnull()]
hist_votes_df['PartyName'] = hist_votes_df.PartyName.apply(lambda p: party_names_translation[p])

#hist_votes_df = hist_votes_df.merge(hist_mla_ids[['PersonId','normal_name','session_name']], on='PersonId', how='inner')
hist_votes_df['DivisionDate'] = pd.to_datetime(hist_votes_df['DivisionDate'], utc=True)
hist_votes_df = hist_votes_df.sort_values('DivisionDate')
#now simplify to print nicer
hist_votes_df['DivisionDate'] = hist_votes_df['DivisionDate'].dt.date
#To pass all votes list, create a column with motion title and url 
#  joined by | so that I can split ont this inside the datatable
hist_votes_df['motion_plus_url'] = hist_votes_df.apply(
    lambda row: f"{row['Title']}|https://aims.niassembly.gov.uk/plenary/details.aspx?&ses=0&doc={row['DocumentID']}&pn=0&sid=vd", axis=1)

#Hist votes PCA
hist_votes_df['vote_num'] = hist_votes_df.Vote.apply(lambda v: {'NO':-1, 'AYE':1, 'ABSTAINED':0}[v]) 

hist_votes_pca_res = {}
#Seem to get a few duplicates where someone has two Designations - OK to dedup before pivot
for session_name in hist_votes_df.session_name.unique():
    hist_votes_one_session = hist_votes_df[hist_votes_df.session_name==session_name]
    hist_votes_one_session = hist_votes_one_session[['normal_name','EventId','vote_num']].drop_duplicates().reset_index()

    hist_votes_pca_df = hist_votes_one_session[['normal_name','EventId','vote_num']]\
        .pivot(index='normal_name', columns='EventId', values='vote_num').fillna(0) 
    tmp = hist_votes_one_session.normal_name.value_counts()
    those_in_threshold_pct_votes = tmp[tmp >= hist_votes_one_session.EventId.nunique()*pca_votes_threshold_fraction].index
    hist_votes_pca_df = hist_votes_pca_df[hist_votes_pca_df.index.isin(those_in_threshold_pct_votes)]

    hist_my_pca = PCA(n_components=2, whiten=True)  #doesn't change results but axis units closer to 1
    hist_my_pca.fit(hist_votes_pca_df)
    #print(hist_my_pca.explained_variance_ratio_)
    hist_mlas_2d_rep = pd.DataFrame({'x': [el[0] for el in hist_my_pca.transform(hist_votes_pca_df)],
                                'y': [el[1] for el in hist_my_pca.transform(hist_votes_pca_df)],
                                'normal_name': hist_votes_pca_df.index,
                                'indiv_page_url': ['/individual?mla_name=' + n.replace(' ','+') for n in hist_votes_pca_df.index],
                                'party': [party_names_translation[hist_mla_ids.loc[hist_mla_ids.normal_name == n].PartyName.iloc[0]] for n in hist_votes_pca_df.index]})
    hist_votes_pca_res[session_name] = (hist_mlas_2d_rep, 
        (100*hist_my_pca.explained_variance_ratio_[0], 100*hist_my_pca.explained_variance_ratio_[1]))

#Hist votes party unity 
# hist_votes_party_unity = hist_votes_df.groupby(['session_name','PartyName','EventId'])\
#     .agg({'Vote': lambda v: len(np.unique(v)),'normal_name': len}).reset_index()\
#     .query('normal_name >= 5')\
#     .groupby(['session_name','PartyName']).agg({'Vote': lambda v: 100*np.mean(v==1), 'EventId': len}).reset_index()
# hist_votes_party_unity.columns = ['session_name','PartyName', 'Percent voting as one', 'n_votes']
# hist_votes_party_unity['tooltip_text'] = hist_votes_party_unity.apply(
#     lambda row: f"{row['PartyName']} voted as one in {row['Percent voting as one']:.0f}% of {row['n_votes']} votes",
#     axis=1
# )

#hist votes commentary - can do all at once and filter later by column vote_date
hist_v_comms = pd.concat([
    feather.read_dataframe(data_dir + 'historical_division_votes_v_comms.feather'),
    feather.read_dataframe(data_dir + 'historical_20202022_division_votes_v_comms.feather')
    ])
print(f'{len(hist_v_comms)} rows in hist votes comms, {sum(hist_v_comms.session_name.notnull())} with session_name')
print('Done historical votes')

hist_plenary_contribs_df = pd.concat([
    pd.read_csv(data_dir + 'hist_lda_scored_plenary_contribs.csv'),
    pd.read_csv(data_dir + 'hist_20202022_lda_scored_plenary_contribs.csv')
    ])
print(f'{len(hist_plenary_contribs_df)} rows in hist plenary contribs')
hist_plenary_contribs_df['session_name'] = hist_plenary_contribs_df.PlenaryDate.apply(assign_session_name)

hist_plenary_contribs_topic_counts =  hist_plenary_contribs_df\
    .groupby(['topic_name','session_name']).count().reset_index()\
    [['topic_name','session_name','session_id']].rename(index=str, columns={'session_id': 'n_contribs'})
hist_plenary_contribs_topic_counts = hist_plenary_contribs_topic_counts.merge(
    pd.DataFrame(lda_top5s, columns=['topic_name','top5_words'])
)
hist_plenary_contribs_topic_counts['tooltip_text'] = hist_plenary_contribs_topic_counts.apply(
    lambda row: f"{row['topic_name']}: strongest words are {row['top5_words']}", axis=1
)

hist_emotions_df = pd.concat([
    feather.read_dataframe(data_dir + 'hist_plenary_hansard_contribs_emotions_averaged.feather'),
    feather.read_dataframe(data_dir + 'hist_20202022_plenary_hansard_contribs_emotions_averaged.feather')
    ])
print(f'{len(hist_emotions_df)} rows in hist emotions')
hist_emotions_df = hist_emotions_df.merge(hist_mla_ids[['normal_name','PartyName','session_name']], 
    left_on=['speaker','session_name'], 
    right_on=['normal_name','session_name'], 
    how='inner')
hist_emotions_party_agg = hist_emotions_df.groupby(['session_name','PartyName','emotion_type']).apply(
    lambda x_grouped: np.average(x_grouped['ave_emotion'], weights=x_grouped['word_count']))\
    .reset_index().rename(index=str, columns={0:'ave_emotion'})

print('Done historical contributions')

#News 
#----
news_df = pd.concat([
    #feather.read_dataframe(data_dir + 'newscatcher_articles_slim_w_sentiment_julaugsep2020.feather'),
    feather.read_dataframe(data_dir + 'newscatcher_articles_slim_w_sentiment_sep2020topresent.feather')
]).drop_duplicates()
news_df = news_df[news_df.published_date >= '2023-12-01'].copy()

# Only really need any that will appear in the top sources plot.
with open(data_dir + 'news_source_pprint_dict.json', 'r') as f:
    news_source_pprint_dict = json.load(f)
news_df['source'] = news_df.source.apply(lambda s: news_source_pprint_dict.get(s, s))

#
news_df['published_date'] = pd.to_datetime(news_df['published_date'])
news_df['published_date_week'] = news_df.published_date.dt.isocalendar().week
#news_df['published_date_week'] = news_df.published_date.dt.week
#early Jan can be counted as week 53 by pd.week - messes things up
news_df.loc[(news_df.published_date_week==53) & (news_df.published_date.dt.day <= 7), 'published_date_week'] = 1
news_df['published_date_year'] = news_df.published_date.dt.year
if len(news_df) > 0:
    news_df['published_date_yweek'] = news_df.apply(
        lambda row: '{:04g}-{:02g}'.format(row['published_date_year'], row['published_date_week']), axis=1)
else:
    news_df['published_date_yweek'] = []

news_df = news_df.merge(mla_ids[['normal_name','PartyName','PartyGroup']],
    how = 'inner', on = 'normal_name')
#drop the first and last weeks which could be partial
#news_df = news_df[(news_df.published_date_week > news_df.published_date_week.min()) &
#    (news_df.published_date_week < news_df.published_date_week.max())]
news_df = news_df.sort_values('published_date', ascending=False)
news_df['date_pretty'] = pd.to_datetime(news_df.published_date).dt.strftime('%Y-%m-%d')
if len(news_df) > 0:
    news_df['title_plus_url'] = news_df.apply(lambda row: f"{row['title']}|{row['link']}", axis=1)
else:
    news_df['title_plus_url'] = []

#Filter to most recent month
news_sources = news_df[news_df.published_date.dt.date > news_df.published_date.dt.date.max()-datetime.timedelta(days=30)][['link','source','PartyGroup']]\
    .drop_duplicates()\
    .groupby(['source','PartyGroup']).link.count().reset_index()\
    .rename(index=str, columns={'link':'News articles'})\
    .sort_values('News articles', ascending=False)
#(now filtering to top 10/15 below in function)
if len(news_sources) > 0:
    news_sources['tooltip_text'] = news_sources.apply(
        lambda row: f"{row['source']}: {row['News articles']} article mention{'s' if row['News articles'] != 1 else ''} of {row['PartyGroup'].lower()}s", 
        axis=1)
else:
    news_sources['tooltip_text'] = []

#dedup articles by party before calculating averages by party - doesn't make a big difference
news_sentiment_by_party_week = news_df[['published_date_year','published_date_week','link','PartyName','sr_sentiment_score']].drop_duplicates()\
    .groupby(['published_date_year','published_date_week','PartyName'])\
    .agg({'link': len, 'sr_sentiment_score': np.nanmean}).reset_index()
#news_sentiment_by_party_week = news_sentiment_by_party_week.sort_values(['PartyName','published_date_week'], ignore_index=True)
#news_sentiment_by_party_week = news_sentiment_by_party_week[news_sentiment_by_party_week.sr_sentiment_score.notnull()]
#news_sentiment_by_party_week = news_sentiment_by_party_week[news_sentiment_by_party_week.url >= 3]  #OK to keep in now because using smoothing
news_sentiment_by_party_week = news_sentiment_by_party_week[news_sentiment_by_party_week.PartyName.isin(
    ['DUP','Alliance','Sinn Fein','UUP','SDLP'])]
#fill in missing weeks before averaging - works for volume only
uniques = news_sentiment_by_party_week[['published_date_year','published_date_week']].drop_duplicates()
uniques = pd.concat([uniques.assign(PartyName = p) for p in news_sentiment_by_party_week.PartyName.unique()]).reset_index(drop=True)
news_sentiment_by_party_week = uniques.merge(news_sentiment_by_party_week, on=['published_date_year','published_date_week','PartyName'], how='left')
#keep missing sentiment weeks as NA but can fill volumes as zero
news_sentiment_by_party_week['link'] = news_sentiment_by_party_week['link'].fillna(0)

news_sentiment_by_party_week = news_sentiment_by_party_week.join(
    news_sentiment_by_party_week.groupby('PartyName', sort=False).link\
        .rolling(news_volume_average_window_weeks, min_periods=1, center=True).mean().reset_index(0),  #the 0 is vital here
    rsuffix='_smooth').rename(index=str, columns={'link_smooth':'vol_smooth'})
#print(news_sentiment_by_party_week.head())
#drop first and last weeks here instead, so that table still shows the most recent articles
news_sentiment_by_party_week['yearweekcomb'] = news_sentiment_by_party_week['published_date_year'] + news_sentiment_by_party_week['published_date_week']
news_sentiment_by_party_week = news_sentiment_by_party_week[(news_sentiment_by_party_week.yearweekcomb > news_sentiment_by_party_week.yearweekcomb.min()) &
    (news_sentiment_by_party_week.yearweekcomb < news_sentiment_by_party_week.yearweekcomb.max())]
#This is used for the boxplot - need mean value by party for tooltip
news_sentiment_by_party_week = news_sentiment_by_party_week.merge(
    news_sentiment_by_party_week.groupby('PartyName').agg( 
        tooltip_text = pd.NamedAgg('sr_sentiment_score', lambda s: f"Mean sentiment score = {np.mean(s):.3f}")
    ), on = 'PartyName', how = 'inner')
#news_sentiment_by_party_week['tooltip_text'] = news_sentiment_by_party_week.apply(
#    lambda row: f"{row['PartyName']}: {row['link']:g} articles in week (avg = {row['vol_smooth']:.1f})", axis=1
#)

# News summaries
# Read from file, remove rows with missing news_coverage_summary, take the latest create_date per normal_name
news_summaries = (feather.read_dataframe(data_dir + 'news_summaries.feather')
                  #.dropna(subset=['news_coverage_summary'])
                  .sort_values('create_date')
                  .drop_duplicates(subset='normal_name', keep='last')
                  .sort_values('normal_name'))
# Keep active politicians only - left join news_summaries to mla_ids
news_summaries = pd.merge(news_summaries, mla_ids[['normal_name']], on='normal_name', how='left')
news_summaries['news_coverage_summary'] = news_summaries.news_coverage_summary.fillna('NONE')
print('Done news')

#Polls 
#-----
elections_df = pd.merge(pd.read_csv(data_dir + 'election_details.csv'),
    pd.read_csv(data_dir + 'election_results.csv'),
    on = 'election_id', how = 'inner'
)
elections_df['date'] = pd.to_datetime(elections_df.date)
elections_df = elections_df.sort_values(['date','party'], ascending=[False,True])
elections_df['date_year'] = elections_df.date.dt.strftime('%Y')
elections_df = elections_df[elections_df.date > pd.to_datetime('2015-01-01')]
if len(elections_df) > 0:
    elections_df['tooltip_text'] = elections_df.apply(
        lambda row: f"{row['party']}: {row['pct']:g}% (election; {row['date_year']} {row['election_type']})", axis=1
    )
else:
    elections_df['tooltip_text'] = []

polls_df = pd.merge(pd.read_csv(data_dir + 'poll_details.csv'),
    pd.read_csv(data_dir + 'poll_results.csv'),
    on = 'poll_id', how = 'inner'
)
polls_df = polls_df[polls_df.party != 'Other']  #no point including Other
polls_df['date'] = pd.to_datetime(polls_df.date)
polls_df = polls_df.sort_values(['date','party'], ascending=[False,True])
polls_df['date_pretty'] = polls_df.date.dt.strftime('%Y-%m-%d')
polls_df = polls_df[polls_df.date > pd.to_datetime('2015-01-01')]
if len(polls_df) > 0:
    polls_df['tooltip_text'] = polls_df.apply(
        lambda row: f"{row['party']}: {row['pct']:g}% (poll; {row['organisation']}, n={row['sample_size']:.0f})", axis=1
    )
else:
    polls_df['tooltip_text'] = []

#Calculate poll averages - now uses both polls and elections
def get_current_avg_poll_pct(polls_df, elections_df, party, current_dt, 
    time_power = 4, 
    assembly_equiv_sample_size = 50000, 
    general_equiv_sample_size = 500):
    local_equiv_sample_size = np.exp((np.log(assembly_equiv_sample_size)+np.log(general_equiv_sample_size))/2)
    
    past_party_elections = elections_df[(elections_df.pct.notnull()) & 
        (elections_df.date < current_dt) & (elections_df.party == party)].copy()
    if past_party_elections.shape[0] > 0:
        past_party_elections['equiv_sample_size'] = past_party_elections.apply(
            lambda row: assembly_equiv_sample_size if row['election_type']=='Assembly' else \
                (general_equiv_sample_size if row['election_type']=='General' else local_equiv_sample_size),
            axis=1
        )
    else:
        past_party_elections['equiv_sample_size'] = []
    
    past_party_polls = polls_df[(polls_df.pct.notnull()) & 
        (polls_df.date < current_dt) & (polls_df.party == party)].copy()
    past_party_polls['equiv_sample_size'] = past_party_polls['sample_size']

    past_party_points = pd.concat([
        past_party_polls[['date','party','pct','equiv_sample_size']],
        past_party_elections[['date','party','pct','equiv_sample_size']]
    ])

    if past_party_points.shape[0] == 0:
        return np.nan
    else:
        past_party_points['poll_age_days'] = (current_dt - past_party_points.date).apply(lambda dt: dt.days)
        past_party_points['wt_factor'] = past_party_points.apply(
            lambda row: 0 if row['poll_age_days'] > 1000 else (row['equiv_sample_size']**0.5) * ((1000-row['poll_age_days'])/1000)**time_power,
            axis=1
        )
        if all(past_party_points['wt_factor'] == 0):
            return np.nan
        else:
            return np.average(past_party_points.pct, weights=past_party_points.wt_factor)

poll_avgs_track = []
earliest_poll_or_election_date = min(polls_df.date.min(), elections_df.date.min())
poll_track_date_range = [datetime.datetime.today() - pd.to_timedelta(i, unit='day') \
    for i in range(0, (datetime.datetime.today() - earliest_poll_or_election_date).days+50, poll_track_timestep)]
for party in polls_df[polls_df.party != 'Other'].party.unique():
    poll_avgs_track.append(pd.DataFrame({'party': party,
        'date': poll_track_date_range,
        'pred_pct': [get_current_avg_poll_pct(polls_df, elections_df, party, d) for d in poll_track_date_range]}))
poll_avgs_track = pd.concat(poll_avgs_track)
print('Done polls')

#Blog articles list
blog_pieces = pd.read_csv(data_dir + 'blog_pieces_list.psv', sep='|').iloc[-1::-1]
blog_pieces = blog_pieces[blog_pieces.date_added <= datetime.date.today().strftime('%Y-%m-%d')]
print('Done blog')

#Postcode stuff
postcodes_to_constits = pd.read_csv(data_dir + 'all_ni_postcode_constits_from_doogal.csv', index_col=None)

combined_demog_table = pd.read_csv(data_dir + 'combined_demographics_out.csv')
#These are in increasing order, i.e. lowest age gets age_rank_order=1

#Election forecast
#-----------------
if show_election_forecast:
    #2022 used 03may_ulivtmadjust
    elct_files_date_string = '03may_ulivtmadjust'
    print('Using election forecast from',elct_files_date_string)
    elct_fcst_cw_fps = pd.read_csv(f'{data_dir}/election_forecast_out_1_cw_first_prefs_{elct_files_date_string}_cw.csv')
    elct_fcst_ens_party_seats = pd.read_csv(f'{data_dir}/election_forecast_out_2_ens_party_seats_{elct_files_date_string}_cw.csv')
    #out_4 is read in below
    elct_fcst_cands_summary = pd.read_csv(f'{data_dir}/election_forecast_out_3_cands_summary_{elct_files_date_string}_cw.csv')
    elct_fcst_seat_deltas = pd.read_csv(f'{data_dir}/election_forecast_out_5_party_constit_seat_deltas_{elct_files_date_string}_cw.csv')

    elct_fcst_cw_fps['party_short'] = elct_fcst_cw_fps.party_short.apply(lambda p: 'Ind.' if p=='Independent' else p)

    biggest_party_fracs = elct_fcst_ens_party_seats.sort_values(['it','n_seats'],ascending=False).groupby('it').apply(
        lambda row: 'Tie' if row.iloc[0].n_seats==row.iloc[1].n_seats else row.iloc[0].party_short
        ).value_counts(normalize=True, sort=True).reset_index().rename(columns={'index': 'party', 0: 'frac_biggest'})
    biggest_party_fracs['tooltip_text'] = biggest_party_fracs.apply(
        lambda row: f"{row['party']} have the most seats in {100*row['frac_biggest']:.0f}% of simulations" if row['party'] != 'Tie' \
        else f"Two parties tie for most seats in {100*row['frac_biggest']:.0f}% of simulations", axis=1)

    def convert_fraction_to_words(fraction):
        if fraction > 0.95:
            return 'an almost certain chance'
        elif fraction > 0.86:
            return 'a 9 in 10 chance'
        elif fraction > 0.82:
            return 'a 5 in 6 chance'
        elif fraction > 0.77:
            return 'a 4 in 5 chance'
        elif fraction > 0.73:
            return 'a 3 in 4 chance'
        elif fraction > 0.68:
            return 'a 7 in 10 chance'
        elif fraction > 0.63:
            return 'a 2 in 3 chance'
        elif fraction > 0.55:
            return 'a 3 in 5 chance'
        elif fraction > 0.45:
            return 'a 1 in 2 chance'
        elif fraction > 0.37:
            return 'a 2 in 5 chance'
        elif fraction > 0.31:
            return 'a 1 in 3 chance'
        elif fraction > 0.27:
            return 'a 3 in 10 chance'
        elif fraction > 0.23:
            return 'a 1 in 4 chance'
        elif fraction > 0.18:
            return 'a 1 in 5 chance'
        elif fraction > 0.14:
            return 'a 1 in 6 chance'
        elif fraction > 0.11:
            return 'a 1 in 8 chance'
        elif fraction > 0.08:
            return 'a 1 in 10 chance'
        elif fraction > 0.06:
            return 'a 1 in 15 chance'
        elif fraction > 0.03:
            return 'a 1 in 20 chance'
        else:
            return 'a <1 in 30 chance'

    def make_prob_string(row):
        res = []
        #don't show 0 seat probs?
        for i in range(1,5):
            if row['party_frac_seats_'+str(i)] > 0:
                res.append((convert_fraction_to_words(row['party_frac_seats_'+str(i)]),
                    row['party_frac_seats_'+str(i)], i))
        res.sort(key = lambda t: t[1], reverse=True)
        res = [f"<b>{t[0]}</b> of getting {t[2]} seat{'s' if t[2] != 1 else ''}" for t in res]
        if len(res) == 0:
            return '<li>no realistic chance of getting any seats</li>'
        elif len(res) == 1:
            return '<li>' + res[0] + '</li>'
        else:        
            return '<li>' + '</li><li>'.join(res) + '</li>'  #if doing sublist, don't need an 'and'

    #Describe the possible party seat outcomes in words
    elct_fcst_cands_summary['party_seats_string'] = elct_fcst_cands_summary.apply(make_prob_string, axis=1)
    #Pretty print party pct changes from last time
    elct_fcst_cands_summary['delta_party_fp_pct_pprint'] = elct_fcst_cands_summary.delta_party_fp_pct.apply(lambda p: f"{p:+.1f}%")
    elct_fcst_cands_summary['delta_party_fp_pct_pprint'][
        (elct_fcst_cands_summary.delta_party_fp_pct == elct_fcst_cands_summary.party_mean_fp_pct) | pd.isnull(elct_fcst_cands_summary.delta_party_fp_pct)] = 'n/a'

#Totals for front page
#---------------------
n_politicians_current_session = mla_ids.shape[0]
rank_split_points = [10, n_politicians_current_session*0.3, n_politicians_current_session*0.7]

n_active_mlas = (mla_ids.role=='MLA').sum()
n_active_mlas_excl_ministers = n_active_mlas - len(mla_minister_roles.keys())

file_change_times = [os.path.getmtime(x) for x in \
    [data_dir + 'tweets_slim_4jun2021_to_present.feather',
     data_dir + 'diary_future_events.psv',
     data_dir + 'newscatcher_articles_slim_w_sentiment_sep2020topresent.feather']]
last_updated_date = time.strftime('%A, %-d %B', time.localtime(max(file_change_times)))

totals_dict = {
    'n_politicians': f"{pd.concat([mla_ids[['normal_name']], hist_mla_ids[['normal_name']]]).normal_name.nunique():,}",
    'n_questions': f"{pd.concat([questions_df, hist_questions_df]).DocumentId.nunique():,}",
    'n_answers': f"{pd.concat([answers_df, hist_answers_df]).DocumentId.nunique():,}",
    'n_votes': f"{pd.concat([votes_df, hist_votes_df]).EventId.nunique():,}",
    'n_contributions': f"{pd.concat([scored_plenary_contribs_df, hist_plenary_contribs_df]).shape[0]:,}",
    'n_tweets': f"{tweets_df.status_id.nunique():,}",
    'n_news': f"{news_df.link.nunique():,}",
    'n_polls': f"{polls_df.poll_id.nunique():,}",
    'last_updated_date': last_updated_date
}

#Tidy up
del answers_df, hist_answers_df, hist_questions_df, vote_results_df, hist_vote_results_df, hist_emotions_df

#----
# Helper functions

def find_profile_photo(person_is_mla, person_id, normal_name, member_other_photo_links, mp_api_number=None):
    if person_is_mla:
        image_url = f"https://aims.niassembly.gov.uk/images/mla/{person_id}_s.jpg"
    elif mp_api_number is not None:
        # Check if Parliament has a portrait for the MP; if not use the thumbnail, which is sometimes present but blank
        parliament_portrait_url = f"https://members-api.parliament.uk/api/Members/{mp_api_number:s}/Portrait?cropType=ThreeFour"
        parliament_thumbnail_url = f"https://members-api.parliament.uk/api/Members/{mp_api_number:s}/Thumbnail"
        if requests.get(parliament_portrait_url).status_code == 200:
            image_url = parliament_portrait_url
        elif requests.get(parliament_thumbnail_url).status_code == 200:
            image_url = parliament_thumbnail_url
        elif normal_name in member_other_photo_links.keys():
            image_url = member_other_photo_links[normal_name]
        else:
            image_url = '#'
    else:
        image_url = '#'
    return image_url

#Plots are not pages themselves
def add_grey_legend(plot, orient='top-right', columns=1, mobile_mode=False):
    return plot.configure_legend(
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding = 7 if mobile_mode else 10,
        cornerRadius=10,
        orient = orient,
        columns = columns,
        labelFontSize = 8 if mobile_mode else 11,
        symbolSize = 40 if mobile_mode else 70,
        titleFontSize = 9 if mobile_mode else 12
    )

# ----
# App routes

@app.before_request
def clear_trailing():
    rp = request.path 
    if rp != '/' and rp.endswith('/'):
        return redirect(rp[:-1])

@app.route('/', methods=['GET'])
def index():
    blog_pieces['background_image_path'] = blog_pieces.background_image_file.apply(lambda f: url_for('static', filename=f))
    return render_template('index.html',
        totals_dict = totals_dict,
        full_mla_list = sorted(mla_ids.normal_name.tolist()),
        postcodes_list = sorted(postcodes_to_constits.Postcode.tolist() + mla_ids.ConstituencyName.unique().tolist()),
        blog_pieces = blog_pieces[:3])

@app.route('/what-they-say', methods=['GET'])
def twitter():
    return render_template('twitter.html',
        full_mla_list = sorted(mla_ids.normal_name.tolist()),
        postcodes_list = sorted(postcodes_to_constits.Postcode.tolist() + mla_ids.ConstituencyName.unique().tolist()),
        info_centr_top5 = tweets_network_top5s['info_centr'].tolist(),
        page_rank_top5 = tweets_network_top5s['page_rank'].tolist(),
        betw_centr_top5 = tweets_network_top5s['betw_centr'].tolist(),
        gen_tweets_dup = gen_tweets.loc[gen_tweets.mla_party=='DUP', 'generated_text'].sample(100).tolist(),
        gen_tweets_uup = gen_tweets.loc[gen_tweets.mla_party=='UUP', 'generated_text'].sample(100).tolist(),
        gen_tweets_alli = gen_tweets.loc[gen_tweets.mla_party=='Alliance', 'generated_text'].sample(100).tolist(),
        gen_tweets_sdlp = gen_tweets.loc[gen_tweets.mla_party=='SDLP', 'generated_text'].sample(100).tolist(),
        gen_tweets_sf = gen_tweets.loc[gen_tweets.mla_party=='Sinn Fein', 'generated_text'].sample(100).tolist())        

@app.route('/what-they-do', methods=['GET'])
def assembly():
    args = request.args
    if 'assembly_session' in args:
        session_to_plot = args.get('assembly_session')
    else:
        session_to_plot = CURRENT_ASSEMBLY_SESSION

    #Exclude events that have now happened
    #diary_df_filtered = diary_df[diary_df['EventDate'] >= datetime.date.today().strftime('%Y-%m-%d')]
    diary_df_filtered = diary_df[diary_df['EndTime'] >= datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')]
    #But not more than 6 items, so as not to clutter page
    diary_df_filtered = diary_df_filtered.head(6)

    if session_to_plot in [CURRENT_ASSEMBLY_SESSION, '']:
        n_mlas = mlas_2d_rep.shape[0]
        n_votes = votes_df.EventId.nunique()
        v_comms_tmp = v_comms.copy()
        committee_meetings_list = [e[1].values.tolist() for e in committee_agendas_agg.head(100).iterrows()]
    else:
        n_mlas = hist_votes_pca_res[session_to_plot][0].shape[0]
        n_votes = hist_votes_df[hist_votes_df.session_name==session_to_plot].EventId.nunique()
        v_comms_tmp = hist_v_comms[hist_v_comms.session_name==session_to_plot].copy()
        committee_meetings_list = []

    votes_list = [e[1].values.tolist() for e in v_comms_tmp[['vote_date','vote_subject','vote_tabler_group','vote_result',
        'uni_bloc_vote','nat_bloc_vote','alli_vote','green_vote','uni_nat_split']].iterrows()]
    chances_to_take_a_side = v_comms_tmp[(v_comms_tmp.uni_nat_split=='Yes')].shape[0]
    alli_num_votes_with_uni = v_comms_tmp[(v_comms_tmp.uni_nat_split=='Yes') &
        (v_comms_tmp.alli_vote==v_comms_tmp.uni_bloc_vote)].shape[0]
    alli_num_votes_with_nat = v_comms_tmp[(v_comms_tmp.uni_nat_split=='Yes') & 
        (v_comms_tmp.alli_vote==v_comms_tmp.nat_bloc_vote)].shape[0]
    green_num_votes_with_uni = v_comms_tmp[(v_comms_tmp.uni_nat_split=='Yes') &
        (v_comms_tmp.green_vote==v_comms_tmp.uni_bloc_vote)].shape[0]
    green_num_votes_with_nat = v_comms_tmp[(v_comms_tmp.uni_nat_split=='Yes') & 
        (v_comms_tmp.green_vote==v_comms_tmp.nat_bloc_vote)].shape[0]

    return render_template('assembly.html',
        session_to_plot = session_to_plot,
        current_session = CURRENT_ASSEMBLY_SESSION,
        session_names_list = valid_session_names,
        diary = diary_df_filtered, 
        n_mlas = n_mlas, 
        n_votes = n_votes,
        full_mla_list = sorted(mla_ids.normal_name.tolist()),
        postcodes_list = sorted(postcodes_to_constits.Postcode.tolist() + mla_ids.ConstituencyName.unique().tolist()),
        votes_list = votes_list,
        pca_votes_threshold_pct = int(pca_votes_threshold_fraction*100),
        votes_passed_details = [(v_comms_tmp.vote_result=='PASS').sum(), v_comms_tmp.shape[0], f"{100*(v_comms_tmp.vote_result=='PASS').mean():.0f}%"],
        #f"{(v_comms_tmp.vote_result=='PASS').sum()} / {v_comms_tmp.shape[0]}",
        uni_nat_split_details = [(v_comms_tmp.uni_nat_split=='Yes').sum(), v_comms_tmp.shape[0], f"{100*(v_comms_tmp.uni_nat_split=='Yes').mean():.0f}%"],
        #f"{(v_comms_tmp.uni_nat_split=='Yes').sum()} / {v_comms_tmp.shape[0]}",
        num_uni_nat_split_passes = ((v_comms_tmp.uni_nat_split=='Yes') & (v_comms_tmp.vote_result=='PASS')).sum(),
        uni_tabled_passed_details = [((v_comms_tmp.vote_tabler_group=='Unionist') & (v_comms_tmp.vote_result=='PASS')).sum(),
            (v_comms_tmp.vote_tabler_group=='Unionist').sum(),
            f"{100*(((v_comms_tmp.vote_tabler_group=='Unionist') & (v_comms_tmp.vote_result=='PASS')).sum() / (v_comms_tmp.vote_tabler_group=='Unionist').sum()):.0f}%"],
        #f"{((v_comms_tmp.vote_tabler_group=='Unionist') & (v_comms_tmp.vote_result=='PASS')).sum()} / {(v_comms_tmp.vote_tabler_group=='Unionist').sum()}",
        #nat_tabled_passed_string = f"{((v_comms_tmp.vote_tabler_group=='Nationalist') & (v_comms_tmp.vote_result=='PASS')).sum()} / {(v_comms_tmp.vote_tabler_group=='Nationalist').sum()}",
        nat_tabled_passed_details = [((v_comms_tmp.vote_tabler_group=='Nationalist') & (v_comms_tmp.vote_result=='PASS')).sum(),
            (v_comms_tmp.vote_tabler_group=='Nationalist').sum(),
            f"{100*(((v_comms_tmp.vote_tabler_group=='Nationalist') & (v_comms_tmp.vote_result=='PASS')).sum() / (v_comms_tmp.vote_tabler_group=='Nationalist').sum()):.0f}%"],
        #mix_tabled_passed_string = f"{((v_comms_tmp.vote_tabler_group=='Mixed') & (v_comms_tmp.vote_result=='PASS')).sum()} / {(v_comms_tmp.vote_tabler_group=='Mixed').sum()}",
        mix_tabled_passed_details = [((v_comms_tmp.vote_tabler_group=='Mixed') & (v_comms_tmp.vote_result=='PASS')).sum(),
            (v_comms_tmp.vote_tabler_group=='Mixed').sum(),
            f"{100*(((v_comms_tmp.vote_tabler_group=='Mixed') & (v_comms_tmp.vote_result=='PASS')).sum() / (v_comms_tmp.vote_tabler_group=='Mixed').sum()):.0f}%"],
        alli_like_uni_details = [alli_num_votes_with_uni, chances_to_take_a_side],
        alli_like_nat_details = [alli_num_votes_with_nat, chances_to_take_a_side],
        green_like_uni_details = [green_num_votes_with_uni, chances_to_take_a_side],
        green_like_nat_details = [green_num_votes_with_nat, chances_to_take_a_side],
        committee_meetings_list = committee_meetings_list)


@app.route('/what-we-report', methods=['GET'])
def news():
    # Collapse to one row per article
    news_df_dedup = (news_df
                     .assign(surname = lambda df: df.apply(lambda row: row['normal_name'].split(' ')[-1], axis=1))
                     .sort_values('surname', ascending=True)
                     .groupby(['date_pretty', 'title_plus_url', 'source'], as_index=False)
                     .agg(normal_names = ('normal_name', lambda l: ', '.join(l)))
                     .sort_values('date_pretty', ascending=False)
                    )

    if test_mode:
        articles_list = [e[1].values.tolist() for e in news_df_dedup.head(50)[['date_pretty', 'title_plus_url', 'source', 'normal_names']].iterrows()]
    else:
        articles_list = [e[1].values.tolist() for e in news_df_dedup.head(1000)[['date_pretty', 'title_plus_url', 'source', 'normal_names']].iterrows()]

    return render_template('news.html',
        articles_list = articles_list,
        full_mla_list = sorted(mla_ids.normal_name.tolist()),
        postcodes_list = sorted(postcodes_to_constits.Postcode.tolist() + mla_ids.ConstituencyName.unique().tolist()),
        news_volume_average_window_weeks = news_volume_average_window_weeks,
        news_summary_politicians = ['-- Please select --'] + news_summaries.normal_name.tolist(),
        news_summary_n_articles = [0] + news_summaries.n_articles.tolist(),
        news_summary_time_period = [''] + news_summaries.time_period.tolist(),
        news_summary_summaries = [''] + news_summaries.news_coverage_summary.tolist())

@app.route('/how-we-vote', methods=['GET'])
def polls():
    polls_tmp = polls_df.loc[polls_df.pct.notnull()].copy()
    polls_tmp['date_plus_url'] = polls_tmp.apply(
        lambda row: f"{row['date_pretty']}|{row['link']}", axis=1
    )
    polls_tmp = polls_tmp[['date_plus_url','organisation','sample_size','party','pct']]

    if show_election_forecast:
        elct_cands_tmp = elct_fcst_cands_summary[['fullname','frac_elected','constit','party_short']].copy()
        elct_cands_tmp['constit'] = elct_cands_tmp.apply(
            lambda row: f"{row['constit']}|postcode?postcode_choice={row['constit'].upper().replace(' ','+')}#election", axis=1)

        return render_template('polls.html',
            poll_results_list = [e[1].values.tolist() for e in polls_tmp.iterrows()],
            full_mla_list = sorted(mla_ids.normal_name.tolist()),
            postcodes_list = sorted(postcodes_to_constits.Postcode.tolist() + mla_ids.ConstituencyName.unique().tolist()),
            elct_all_cand_list = [e[1].values.tolist() for e in elct_cands_tmp.iterrows()],
            elct_n_ensemble = elct_fcst_ens_party_seats.it.max())
    else:
        return render_template('polls.html',
            poll_results_list = [e[1].values.tolist() for e in polls_tmp.iterrows()],
            full_mla_list = sorted(mla_ids.normal_name.tolist()),
            postcodes_list = sorted(postcodes_to_constits.Postcode.tolist() + mla_ids.ConstituencyName.unique().tolist()))

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html',
        full_mla_list = sorted(mla_ids.normal_name.tolist()),
        postcodes_list = sorted(postcodes_to_constits.Postcode.tolist() + mla_ids.ConstituencyName.unique().tolist()))

@app.route('/blog', methods=['GET'])
def blog():
    blog_pieces['background_image_path'] = blog_pieces.background_image_file.apply(lambda f: url_for('static', filename=f))
    return render_template('blog.html', 
        full_mla_list = sorted(mla_ids.normal_name.tolist()),
        postcodes_list = sorted(postcodes_to_constits.Postcode.tolist() + mla_ids.ConstituencyName.unique().tolist()),
        blog_pieces = blog_pieces)

@app.route('/blog/<post_name>', methods=['GET'])
def blog_item(post_name):
    blog_links = blog_pieces['link'].tolist()
    blog_titles = blog_pieces['title'].tolist()

    if post_name in blog_links:
        place_in_blog_list = blog_links.index(post_name)

        # Pass up to 3 dfs to use as tables in the blog post;
        #   only some post_names use any tables.
        df_1_list, df_2_list, df_3_list = None, None, None
        if post_name in ['odni-minister-data-1',
                         'odni-minister-data-2',
                         'party-funding']:
            df_1 = pd.read_csv(data_dir + f'blog-data_{post_name}_table-1.csv')
            df_1_list = [e[1].values.tolist() for e in df_1.iterrows()]
        if post_name in ['odni-minister-data-1',
                         'odni-minister-data-2']:
            df_2 = pd.read_csv(data_dir + f'blog-data_{post_name}_table-2.csv')
            df_2_list = [e[1].values.tolist() for e in df_2.iterrows()]
        if post_name in ['odni-minister-data-1']:
            df_3 = pd.read_csv(data_dir + f'blog-data_{post_name}_table-3.csv')
            df_3_list = [e[1].values.tolist() for e in df_3.iterrows()]

        return render_template('blog-'+post_name+'.html',
            full_mla_list = sorted(mla_ids.normal_name.tolist()),
            postcodes_list = sorted(postcodes_to_constits.Postcode.tolist() + mla_ids.ConstituencyName.unique().tolist()),
            prev_and_next_article_title = (
                None if place_in_blog_list==blog_pieces.shape[0]-1 else blog_titles[place_in_blog_list+1],
                None if place_in_blog_list==0 else blog_titles[place_in_blog_list-1]),
            prev_and_next_article_link = (
                None if place_in_blog_list==blog_pieces.shape[0]-1 else blog_links[place_in_blog_list+1],
                None if place_in_blog_list==0 else blog_links[place_in_blog_list-1]),
            df_1_list = df_1_list,
            df_2_list = df_2_list,
            df_3_list = df_3_list,
            )
    else:
        return blog()        

@app.route('/individual', methods=['GET'])
def indiv():
    args = request.args
    if 'mla_name' in args and args.get('mla_name') != 'Choose MLA...' \
        and args.get('mla_name') in mla_ids.normal_name.unique():
        person_selected = True
        person_choice = args.get('mla_name')                
        person_choice_party = mla_ids[mla_ids.normal_name==person_choice].PartyName_long.iloc[0]
        person_name_string = f"{person_choice}"
        person_name_lc = person_choice.lower()

        row = mla_ids[mla_ids.normal_name == person_choice].iloc[0]

        date_added = row['added']

        if row['role'] in ['MLA','MP']:
            person_name_string = person_name_string + f" {row['role']}"

        person_constit = row['ConstituencyName']

        person_is_mla = row['role'] == 'MLA'
        image_url = find_profile_photo(person_is_mla, row.PersonId, row.normal_name, member_other_photo_links,
            mp_api_number=mp_api_numbers.get(person_choice, None))
        if person_is_mla:
            email_address = mla_ids[mla_ids.PersonId==row.PersonId].AssemblyEmail.iloc[0]
        else:
            email_address = mla_ids[mla_ids.normal_name==person_choice].WestminsterEmail.iloc[0]
        if email_address == 'none':
            email_address = None

        if person_choice in mla_minister_roles.keys():
            person_name_string += f"</br>({mla_minister_roles[person_choice]})"

        if person_is_mla and row['normal_name'] in committee_roles.normal_name.tolist() and row['normal_name'] in committee_meeting_attendance.normal_name.tolist():
            person_committee_roles_raw = committee_roles[committee_roles.normal_name == row['normal_name']].Organisation.tolist()
            person_committee_roles = committee_roles[committee_roles.normal_name == row['normal_name']].apply(
                lambda row: f"{row['Organisation']}{ ' ('+row['Role']+')' if 'Chair' in row['Role'] else ''}", axis=1
            ).tolist()

            # NEW
            person_committee_attendances = (
                committee_meeting_attendance[committee_meeting_attendance.normal_name == row['normal_name']]
                .groupby('committee_name', as_index=False)
                .agg(n_meetings = ('meeting_date', len),
                     n_attended = ('attended', 'sum'),
                     first_date = ('meeting_date', 'min'))
                .assign(first_date = lambda df: df.first_date.dt.strftime('%Y-%m-%d'),
                    summ_string = lambda df: df.apply(lambda row: f'Attended {row.n_attended} of {row.n_meetings} meetings since {row.first_date}', axis=1))
                )
            person_committee_work = []
            for cr, cr_raw in zip(person_committee_roles, person_committee_roles_raw):
                if cr_raw in person_committee_attendances.committee_name.tolist():
                    person_committee_work.append(
                        (cr, person_committee_attendances.query(f"committee_name == '{cr_raw}'").summ_string.item())
                        )
                else:
                    person_committee_work.append((cr, 'No recent meetings found for this committee'))
        else:
            person_committee_work = []

        if person_is_mla and row['normal_name'] in mla_interests.normal_name.tolist():
            person_interests = mla_interests[mla_interests.normal_name==row['normal_name']].apply(
                lambda row: f"{row.RegisterCategory}: {row.RegisterEntry}", axis=1
            ).tolist()
        else:
            person_interests = ["No interests declared."]

        if person_is_mla and row['normal_name'] in allpartygroup_memberships.normal_name.tolist():
            person_apgs = (allpartygroup_memberships[allpartygroup_memberships.normal_name==row['normal_name']]
                .apg_name.unique().tolist()
                )
        else:
            person_apgs = []

        #Better to do last month as I will only be updating weekly
        tweets_last_month = tweets_df[(tweets_df.normal_name==row['normal_name']) &
            (tweets_df.created_at.dt.date > datetime.date.today()-datetime.timedelta(days=30))].shape[0]

        #tweets_by_week = tweets_df[tweets_df.normal_name==row['normal_name']].created_at.dt.week.value_counts().sort_index().tolist()
        tmp = tweets_df[tweets_df.normal_name==row['normal_name']].created_at_yweek.value_counts()
        tmp = pd.DataFrame({'created_at_yweek': tmp.index, 'n_tweets': tmp.values})
        tweets_by_week = tweets_df[['created_at_yweek']].drop_duplicates()\
            .merge(tmp, on='created_at_yweek', how='left')\
            .fillna(0).sort_values('created_at_yweek')['n_tweets'].astype(int).to_list()


        sample_recent_tweets = tweets_df[(tweets_df.normal_name==row['normal_name']) &
            (~tweets_df.is_retweet)].sort_values('created_at', ascending=False).head(15)[['created_at','text','quoted_status_id']]
        if sample_recent_tweets.shape[0] > 0:
            sample_recent_tweets['created_at'] = sample_recent_tweets.created_at.dt.strftime('%Y-%m-%d')
            sample_recent_tweets['involves_quote'] = sample_recent_tweets.quoted_status_id.apply(lambda s: s is not None) 
            sample_recent_tweets['quoted_url'] = sample_recent_tweets.apply(
                lambda row: re.findall('//t.*', row['text'])[0] if row['involves_quote'] else '', axis=1
            )
            sample_recent_tweets['text'] = sample_recent_tweets.text.str.replace('//t.*', '', regex=True)
            sample_recent_tweets = sample_recent_tweets[sample_recent_tweets.text != '']
        if sample_recent_tweets.shape[0] > 5:
            sample_recent_tweets = sample_recent_tweets.sample(5)\
            .sort_values('created_at', ascending=False)

        if sum(tweets_df.normal_name==row['normal_name']) > 0:
            twitter_handle = tweets_df[tweets_df.normal_name==row['normal_name']].screen_name.iloc[-1]

            tweet_volume_rank = tweets_df['normal_name'].value_counts().index.get_loc(row['normal_name']) + 1
            #
            if tweet_volume_rank <= rank_split_points[0]:
                tweet_volume_rank_string = f"Tweets <b>very frequently</b>"
            elif tweet_volume_rank <= rank_split_points[1]:
                tweet_volume_rank_string = f"Tweets <b>fairly frequently</b>"
            elif tweet_volume_rank <= rank_split_points[2]:
                tweet_volume_rank_string = f"Tweets at an <b>average rate</b>"
            else:
                tweet_volume_rank_string = f"<b>Doesn't tweet very often</b>"
            tweet_volume_rank_string += f"<br />(<b>#{tweet_volume_rank} / {n_politicians_current_session}</b> in total tweets since 1 July 2020)"
        else:
            twitter_handle = None
            tweet_volume_rank_string = "We don't know of a Twitter account for this member"
        member_tweet_volumes = tweets_df['normal_name'].value_counts().values.tolist()

        #member_retweets requires having at least 10 tweets in last 5 weeks
        if sum(member_retweets['normal_name'] == row['normal_name']) > 0:
            retweet_rate = member_retweets[member_retweets['normal_name'] == row['normal_name']]['retweets_per_tweet'].iloc[0]
            retweet_rate_rank = (member_retweets['normal_name'] == row['normal_name']).values.argmax()+1
            if retweet_rate_rank <= 10:
                retweet_rate_rank_string = f"<b>High</b> Twitter impact"
            elif retweet_rate_rank <= 0.3*member_retweets.shape[0]:
                retweet_rate_rank_string = f"<b>Fairly high</b> Twitter impact"
            elif retweet_rate_rank <= 0.7*member_retweets.shape[0]:
                retweet_rate_rank_string = f"<b>Average</b> Twitter impact"
            else:
                retweet_rate_rank_string = f"<b>Low</b> Twitter impact"
            retweet_rate_rank_string += f"<br />(<b>#{retweet_rate_rank} / {member_retweets.shape[0]}</b> in retweets per original tweet)"
        else:
            retweet_rate_rank_string = 'n/a'
            retweet_rate = None
        member_retweet_rates = member_retweets['retweets_per_tweet'].tolist()

        if row['normal_name'] in member_tweet_sentiment.normal_name.tolist():
            tweet_positivity = member_tweet_sentiment[member_tweet_sentiment['normal_name'] == row['normal_name']]['sentiment_vader_compound'].iloc[0]
            tweet_positivity_rank = (member_tweet_sentiment['normal_name'] == row['normal_name']).values.argmax()+1
            if tweet_positivity_rank <= 10:
                tweet_positivity_rank_string = f"Tweets <b>very positive</b> messages"
            elif tweet_positivity_rank <= 0.3*member_tweet_sentiment.shape[0]:
                tweet_positivity_rank_string = f"Tweets <b>fairly positive</b> messages"
            elif tweet_positivity_rank <= 0.7*member_tweet_sentiment.shape[0]:
                tweet_positivity_rank_string = f"Tweets messages of <b>average sentiment</b>"
            else:
                tweet_positivity_rank_string = f"Tweets <b>relatively negative</b> messages"
            tweet_positivity_rank_string += f"<br />(<b>#{tweet_positivity_rank} / {member_tweet_sentiment.shape[0]}</b> for tweet positivity)"
        else:
            tweet_positivity_rank_string = 'n/a'
            tweet_positivity = None
        member_tweet_positivities = member_tweet_sentiment['sentiment_vader_compound'].tolist()

        news_articles_last_month = news_df[(news_df.normal_name==row['normal_name']) &
            (news_df.published_date.dt.date > datetime.date.today()-datetime.timedelta(days=30))].shape[0]

        tmp = news_df[news_df.normal_name==row['normal_name']].published_date_yweek.value_counts()
        tmp = pd.DataFrame({'published_date_yweek': tmp.index, 'n_mentions': tmp.values})
        news_articles_by_week = news_df[['published_date_yweek']].drop_duplicates()\
            .merge(tmp, on='published_date_yweek', how='left')\
            .fillna(0).sort_values('published_date_yweek')['n_mentions'].astype(int).to_list()

        if person_choice in news_summaries.normal_name.tolist():
            member_news_summaries = news_summaries[news_summaries.normal_name==person_choice].iloc[0]
            if member_news_summaries.n_articles > 0:
                news_summary_desc_string = (
                    f"(Summary of {member_news_summaries.n_articles} articles "
                    f"in the period from {member_news_summaries.time_period.split('_')[0]} to now)"
                    )
            else:
                news_summary_desc_string = ""
            news_summary_summary = member_news_summaries.news_coverage_summary.replace('\n', '<br/>')
        else:
            # members that have never had a news article will not be in the summaries table at all
            news_summary_desc_string, news_summary_summary = "", ""

        # Assembly votes, questions, plenary
        if person_is_mla:
            mla_votes_list = []
            tmp = votes_df.loc[votes_df['normal_name'] == row['normal_name'], ['DivisionDate','motion_plus_url','Vote']].sort_values('DivisionDate', ascending=False)
            if tmp.shape[0] > 0:
                mla_votes_list = [e[1].values.tolist() for e in tmp.iterrows()]

            #Account for people joining midway through a session
            vote_date_added = '2020-01-11' if date_added == '2020-08-01' else date_added #I tracked most people from 1 Aug 2020 but have their vote info from Jan 2020
            votes_they_joined_before = votes_df[(votes_df.DivisionDate.astype(str) >= vote_date_added) | (votes_df.normal_name==row['normal_name'])]
            votes_present_numbers = (sum(votes_df['normal_name'] == row['normal_name']), votes_they_joined_before.EventId.nunique())
            if len(votes_they_joined_before) > 0:
                votes_present_first_date = votes_they_joined_before.DivisionDate.min().strftime('%d %B %Y')
            else:
                votes_present_first_date = 'joining the Assembly'
            votes_present_string = f"<b>{votes_present_numbers[0]}" + \
                f" / {votes_present_numbers[1]} votes</b> since {votes_present_first_date}"

            num_questions = (questions_df['normal_name'] == row['normal_name']).sum()
            member_question_volumes = questions_df['normal_name'].value_counts().values.tolist()
            if num_questions > 0:
                questions_rank = questions_df['normal_name'].value_counts().index.get_loc(row['normal_name']) + 1
                questions_rank_string = f"<b>#{questions_rank} / {n_active_mlas}</b>"
            else:
                questions_rank_string = f"<b>#{questions_df.normal_name.nunique()+1}-{n_active_mlas} / {n_active_mlas}</b>"
            #can't consistently work out the denominator excluding ministers so just use the 90

            num_plenary_contribs = (scored_plenary_contribs_df['speaker'] == row['normal_name']).sum()
            if num_plenary_contribs > 0:
                plenary_contribs_rank = scored_plenary_contribs_df['speaker'].value_counts().index.get_loc(row['normal_name']) + 1
                plenary_contribs_rank_string = f"<b>#{plenary_contribs_rank} / {max(n_active_mlas, scored_plenary_contribs_df.speaker.nunique())}</b>"
            else:
                plenary_contribs_rank_string = f"<b>#{scored_plenary_contribs_df.speaker.nunique()+1}-{n_active_mlas} / {n_active_mlas}</b>"
            member_contribs_volumes = scored_plenary_contribs_df['speaker'].value_counts().values.tolist()

            top_contrib_topics = scored_plenary_contribs_df[scored_plenary_contribs_df['speaker'] == row['normal_name']]\
                .topic_name.value_counts(normalize=True, dropna=False)
            top_contrib_topics = top_contrib_topics[top_contrib_topics.index != 'misc./none']
            #send topic|pct for font size|color for font
            if len(top_contrib_topics) >= 3:
                top_contrib_topic_list = [f"{top_contrib_topics.index[0]} ({top_contrib_topics.values[0]*100:.0f}%)|{36*max(min(top_contrib_topics.values[0]/0.4,1), 16/36):.0f}|{plenary_contribs_colour_dict[top_contrib_topics.index[0]]}",
                    f"{top_contrib_topics.index[1]} ({top_contrib_topics.values[1]*100:.0f}%)|{36*max(min(top_contrib_topics.values[1]/0.4,1), 16/36):.0f}|{plenary_contribs_colour_dict[top_contrib_topics.index[1]]}",
                    f"{top_contrib_topics.index[2]} ({top_contrib_topics.values[2]*100:.0f}%)|{36*max(min(top_contrib_topics.values[2]/0.4,1), 16/36):.0f}|{plenary_contribs_colour_dict[top_contrib_topics.index[2]]}"
                ]
            else:
                top_contrib_topic_list = []
            #emotions
            member_emotion_ranks_string = "Plenary contributions language scores relatively <b>high</b> on "
            member_any_top_emotion = False
            for emotion_type in ['anger','anticipation','joy','sadness','trust']:
                tmp = emotions_df[(emotions_df.emotion_type==emotion_type) & (emotions_df.word_count >= 100)].sort_values('ave_emotion', ascending=False)
                if row['normal_name'] in tmp.speaker.tolist():
                    member_rank = (tmp.speaker==row['normal_name']).idxmax()+1
                    if member_rank <= 15:
                        member_any_top_emotion = True
                        member_emotion_ranks_string += f"<b>{emotion_type}</b> (#{member_rank}/{tmp.shape[0]}), "
            if member_any_top_emotion:
                member_emotion_ranks_string = member_emotion_ranks_string[:-2]
            else:
                member_emotion_ranks_string = None
                #bit of a simplification because words can score for two emotions, but roughly
                #  estimate total emotion by adding 5 core emotion fractions
                tmp = emotions_df[(emotions_df.emotion_type.isin(['anger','anticipation','joy','sadness','trust','disgust','fear','surprise'])) &
                    (emotions_df.word_count >= 100)]\
                    .groupby('speaker').agg({'ave_emotion': sum}).reset_index().sort_values('ave_emotion', ascending=True) 
                if row['normal_name'] in tmp.speaker.tolist():
                    member_rank = (tmp.speaker==row['normal_name']).idxmax()+1
                    if member_rank <= 20:
                        member_emotion_ranks_string = f"Plenary contributions language scores relatively <b>low on emotion</b> overall (<b>#{tmp.shape[0]-member_rank+1} / {tmp.shape[0]}</b>)"
        else:
            mla_votes_list = None
            votes_present_string = None
            votes_present_numbers = [0, 0]
            num_questions = None
            questions_rank_string = None
            member_question_volumes = None
            num_plenary_contribs = None
            plenary_contribs_rank_string = None
            member_contribs_volumes = None
            top_contrib_topic_list = None
            member_emotion_ranks_string = None
            votes_present_string = None

        return render_template('indiv.html', 
            person_selected = person_selected,
            person_is_mla = person_is_mla,
            mla_or_mp_id = row.PersonId if person_is_mla else mp_api_numbers.get(person_choice, None),
            full_mla_list = sorted(mla_ids.normal_name.tolist()),
            postcodes_list = sorted(postcodes_to_constits.Postcode.tolist() + mla_ids.ConstituencyName.unique().tolist()),
            person_name_string = person_name_string,
            person_name_lc = person_name_lc,
            person_date_added = date_added,
            news_tracked_since_date = 'December 2023', # if date_added < '2023-12-01' else date_added,  # confusing for MLA->MPs
            person_party = person_choice_party,
            person_committee_work = person_committee_work,
            person_interests = person_interests,
            person_apgs = person_apgs,
            image_url = image_url,
            twitter_handle = twitter_handle,
            email_address = email_address,
            tweets_last_month = tweets_last_month, 
            tweets_by_week = tweets_by_week,
            tweet_volume_rank_string = tweet_volume_rank_string,
            member_tweet_volumes = member_tweet_volumes,
            retweet_rate_rank_string = retweet_rate_rank_string,
            retweet_rate = retweet_rate,
            member_retweet_rates = member_retweet_rates,
            tweet_positivity_rank_string = tweet_positivity_rank_string,
            tweet_positivity = tweet_positivity,
            member_tweet_positivities = member_tweet_positivities,
            sample_recent_tweets = sample_recent_tweets,
            news_articles_last_month = news_articles_last_month,
            news_articles_by_week = news_articles_by_week,
            news_summary_desc_string = news_summary_desc_string,
            news_summary_summary = news_summary_summary,
            mla_votes_list = mla_votes_list,
            votes_present_string = votes_present_string,
            votes_present_numbers = votes_present_numbers,
            num_questions = num_questions,
            questions_rank_string = questions_rank_string,
            member_question_volumes = member_question_volumes,
            num_plenary_contribs = num_plenary_contribs,
            plenary_contribs_rank_string = plenary_contribs_rank_string,
            member_contribs_volumes = member_contribs_volumes,
            top_contrib_topic_list = top_contrib_topic_list,
            member_emotion_ranks_string = member_emotion_ranks_string,
            person_constit = person_constit,
            include_twitter = include_twitter)

    else:
        blog_pieces['background_image_path'] = blog_pieces.background_image_file.apply(lambda f: url_for('static', filename=f))
        return render_template('index.html',
            totals_dict = totals_dict,
            full_mla_list = sorted(mla_ids.normal_name.tolist()),
            postcodes_list = sorted(postcodes_to_constits.Postcode.tolist() + mla_ids.ConstituencyName.unique().tolist()),
            blog_pieces = blog_pieces[:3])

@app.route('/postcode', methods=['GET'])
def postcode():
    args = request.args
    if 'postcode_choice' in args:
        postcode_choice = args.get('postcode_choice')
    else:
        postcode_choice = 'BT1 1AA'

    postcode_choice = postcode_choice.upper()
    if postcode_choice in mla_ids.ConstituencyName.str.upper().tolist():
        constit_choice = postcode_choice
        heading_message = f"These are the representatives for the <b>{constit_choice}</b> constituency."
        #Get any postcode from this constit to use for the Write To Them link
        #  (a few of them are no longer found on Write To Them, but unlikely to pick those)
        postcode_choice = postcodes_to_constits[postcodes_to_constits.Constituency.str.upper()==constit_choice].Postcode.sample(1).iloc[0]
    elif postcode_choice in postcodes_to_constits.Postcode.tolist():
        constit_choice = postcodes_to_constits[postcodes_to_constits.Postcode==postcode_choice].Constituency.iloc[0].upper()
        heading_message = f"{postcode_choice} is part of the <b>{constit_choice}</b> constituency."
    elif postcode_choice == '':
        constit_choice = mla_ids.ConstituencyName.str.upper().sample(1).item()
        heading_message = f"No postcode selected. Showing randomly selected constituency <b>{constit_choice}</b>."
    else:
        constit_choice = mla_ids.ConstituencyName.str.upper().sample(1).item()
        heading_message = f"{postcode_choice} not found. Showing randomly selected constituency <b>{constit_choice}</b>."

    mla_choices = mla_ids[(mla_ids.active == 1) & (mla_ids.ConstituencyName.str.upper() == constit_choice)]
    mla_choices = mla_choices.sort_values(['role','MemberLastName'])

    normal_names_list = mla_choices.normal_name.tolist()
    mla_or_mp_ids_list = mla_choices.apply(lambda row: 
        row.PersonId if row.role == 'MLA' else mp_api_numbers.get(row.normal_name, None),
        axis=1).tolist()

    rep_image_urls_list = []
    votes_present_string_list = []
    top_contrib_topic_list_list = []

    for row in mla_choices.itertuples():
        image_url = find_profile_photo(
            row.role == 'MLA',
            row.PersonId,
            row.normal_name,
            member_other_photo_links,
            mp_api_number=mp_api_numbers.get(row.normal_name, None)
            )
        rep_image_urls_list.append(image_url)

        date_added = row.added

        if row.role == 'MLA':
            #Account for people joining midway through a session
            vote_date_added = '2020-01-11' if date_added == '2020-08-01' else date_added #I tracked most people from 1 Aug 2020 but have their vote info from Jan 2020
            votes_they_joined_before = votes_df[(votes_df.DivisionDate.astype(str) >= vote_date_added) | (votes_df.normal_name==row.normal_name)]
            votes_present_numbers = (sum(votes_df['normal_name'] == row.normal_name), votes_they_joined_before.EventId.nunique())
            votes_present_string_list.append(f"<b>{votes_present_numbers[0]}" + \
                f" / {votes_present_numbers[1]} Assembly votes</b> in the current session")
        else:
            votes_present_string_list.append('n/a')

        if row.role == 'MLA':
            top_contrib_topics = scored_plenary_contribs_df[scored_plenary_contribs_df['speaker'] == row.normal_name]\
                .topic_name.value_counts(normalize=True, dropna=False)
            top_contrib_topics = top_contrib_topics[top_contrib_topics.index != 'misc./none']
            #send topic|pct for font size|color for font
            if len(top_contrib_topics) >= 3:
                top_contrib_topic_list = [f"{top_contrib_topics.index[0]} ({top_contrib_topics.values[0]*100:.0f}%)|{36*max(min(top_contrib_topics.values[0]/0.4,1), 16/36):.0f}|{plenary_contribs_colour_dict[top_contrib_topics.index[0]]}",
                    f"{top_contrib_topics.index[1]} ({top_contrib_topics.values[1]*100:.0f}%)|{36*max(min(top_contrib_topics.values[1]/0.4,1), 16/36):.0f}|{plenary_contribs_colour_dict[top_contrib_topics.index[1]]}",
                    f"{top_contrib_topics.index[2]} ({top_contrib_topics.values[2]*100:.0f}%)|{36*max(min(top_contrib_topics.values[2]/0.4,1), 16/36):.0f}|{plenary_contribs_colour_dict[top_contrib_topics.index[2]]}"
                ]
            else:
                top_contrib_topic_list = 'n/a'
            top_contrib_topic_list_list.append(top_contrib_topic_list)
        else:
            top_contrib_topic_list_list.append('n/a')

    rep_twitter_handles_list = [tweets_df[tweets_df.normal_name==nn].screen_name.iloc[-1] if nn in tweets_df.normal_name.unique() 
        else None 
        for nn in normal_names_list]

    rep_email_addrs_list = [row.AssemblyEmail if row.role=='MLA' else row.WestminsterEmail for row in mla_choices.itertuples()]
    rep_email_addrs_list = [None if e == 'none' else e for e in rep_email_addrs_list]

    tweet_volume_rank_string_list = []
    retweet_rate_rank_string_list = []
    for nn in normal_names_list:
        if sum(tweets_df.normal_name==nn) > 0:
            twitter_handle = tweets_df[tweets_df.normal_name==nn].screen_name.iloc[-1]

            tweet_volume_rank = tweets_df['normal_name'].value_counts().index.get_loc(nn) + 1
            #
            if tweet_volume_rank <= rank_split_points[0]:
                tweet_volume_rank_string_list.append(f"Tweets <b>very frequently</b>")
            elif tweet_volume_rank <= rank_split_points[1]:
                tweet_volume_rank_string_list.append(f"Tweets <b>fairly frequently</b>")
            elif tweet_volume_rank <= rank_split_points[2]:
                tweet_volume_rank_string_list.append(f"Tweets at an <b>average rate</b>")
            else:
                tweet_volume_rank_string_list.append(f"Doesn't tweet very often")
        else:
            tweet_volume_rank_string_list.append("We don't know of a Twitter account for this member")

        #member_retweets requires having at least 10 tweets in last 5 weeks
        if sum(member_retweets['normal_name'] == nn) > 0:
            retweet_rate = member_retweets[member_retweets['normal_name'] == nn]['retweets_per_tweet'].iloc[0]
            retweet_rate_rank = (member_retweets['normal_name'] == nn).values.argmax()+1
            if retweet_rate_rank <= 10:
                retweet_rate_rank_string_list.append(f"<b>High</b> Twitter impact")
            elif retweet_rate_rank <= 0.3*member_retweets.shape[0]:
                retweet_rate_rank_string_list.append(f"<b>Fairly high</b> Twitter impact")
            elif retweet_rate_rank <= 0.7*member_retweets.shape[0]:
                retweet_rate_rank_string_list.append(f"<b>Average</b> Twitter impact")
            else:
                retweet_rate_rank_string_list.append(f"<b>Low</b> Twitter impact")
        else:
            retweet_rate_rank_string_list.append('n/a')

    #Demographics

    demogs_row_is_constit = np.where(combined_demog_table.constit.str.upper() == constit_choice)[0][0]
    demogs_for_constit = combined_demog_table.iloc[demogs_row_is_constit]

    young_rank = combined_demog_table['mean_age'].rank().astype(int).iloc[demogs_row_is_constit]
    wage_rank = combined_demog_table['median_wage'].rank(ascending=False).astype(int).iloc[demogs_row_is_constit]
    
    if young_rank <= 6:
        young_rank_text = f'It is the {young_rank}th youngest constituency'
    elif young_rank >= 13:
        young_rank_text = f'It is the {18-young_rank+1}th oldest constituency'
    else:
        young_rank_text = ''
        
    if wage_rank <= 6:
        wage_rank_text = f'has the {wage_rank}th highest median wage'
    elif wage_rank >= 13:
        wage_rank_text = f'has the {18-wage_rank+1}th lowest median wage'
    else:
        wage_rank_text = ''
        
    if young_rank_text != '' and wage_rank_text != '':
        comb_rank_text = ' and '.join([young_rank_text, wage_rank_text]) + '.'
    elif young_rank_text != '':
        comb_rank_text = young_rank_text + '.'
    elif wage_rank_text != '':
        comb_rank_text = 'It '+wage_rank_text+' of the 18 constituencies.'
    else:
        comb_rank_text = ''

    comb_rank_text = comb_rank_text.replace('1th ','').replace('2th','2nd').replace('3th','3rd')

    #Elections
    if show_election_forecast:
        #Read in html table code for candidate results
        with open(f"{data_dir}/election_forecast_out_4_cand_table_{constit_choice.replace(' ','-').lower()}_{elct_files_date_string}_cw.txt", 'r') as f:
            cands_table = f.read()

        elct_fcst_constit_party_stuff = elct_fcst_cands_summary[elct_fcst_cands_summary.constit.str.upper() == constit_choice]
        elct_fcst_constit_party_stuff = elct_fcst_constit_party_stuff[['party_short','party_seats_string','party_mean_fp_pct','delta_party_fp_pct','delta_party_fp_pct_pprint'] + 
            [c for c in elct_fcst_constit_party_stuff.columns if 'frac_seats' in c]].drop_duplicates()
        elct_fcst_constit_party_stuff = elct_fcst_constit_party_stuff.sort_values('party_mean_fp_pct', ascending=False)
        
        elct_fcst_seat_histograms_list = elct_fcst_constit_party_stuff[sorted([c for c in elct_fcst_constit_party_stuff.columns if 'frac_seats' in c])]\
                .fillna(0).apply(lambda row: row.tolist(), axis=1).tolist()
        #Parties with no elected didn't get seat fracs written; keep these as all 0s so we can skip drawing the sparkline on the page

        #Make it a list of lists to pass to page
        elct_fcst_constit_party_stuff = [
            elct_fcst_constit_party_stuff.party_short.tolist(),  #party
            elct_fcst_constit_party_stuff.party_seats_string.tolist(),  #description
            elct_fcst_seat_histograms_list,  #seat probs histogram
            ['grey' if (re.match('Ind',p) or p not in party_colours.party_name.tolist()) else party_colours[party_colours.party_name==p].colour.iloc[0] for p in elct_fcst_constit_party_stuff.party_short],  #colour
            elct_fcst_constit_party_stuff.party_mean_fp_pct.tolist(),  #party fp pct
            elct_fcst_constit_party_stuff.delta_party_fp_pct_pprint.tolist(),  #party fp pct change from last time
            ['lime' if d >= 3 else ('palegreen' if d >= 1 else ('red' if d <= -3 else ('darksalmon' if d <= -1 else 'black')))
                for d in elct_fcst_constit_party_stuff.delta_party_fp_pct],  #display colour for delta fp pct
            [round(1.0 + min(pct,60)*0.9/60, 2) for pct in elct_fcst_constit_party_stuff.party_mean_fp_pct]  #display size for fp pct
        ]
        elct_n_ensemble = elct_fcst_ens_party_seats.it.max()
    else:
        elct_fcst_constit_party_stuff = None
        cands_table = None
        elct_n_ensemble = None
    

    return render_template('postcode.html',
        postcode_choice = postcode_choice,
        constit_choice = constit_choice,
        heading_message = heading_message,
        rep_names_list = normal_names_list,
        rep_parties_list = mla_choices.PartyName_long.tolist(),
        rep_party_colours_list = [party_colours[party_colours.party_name==p].colour.iloc[0] for p in mla_choices.PartyName],
        rep_roles_list = mla_choices.role.tolist(),
        rep_email_addrs_list = rep_email_addrs_list,
        rep_twitter_handles_list = rep_twitter_handles_list,
        rep_mla_or_mps_ids_list = mla_or_mp_ids_list,
        rep_image_urls_list = rep_image_urls_list,
        rep_added_dates_list = mla_choices.added.tolist(),
        tweet_volume_rank_string_list = tweet_volume_rank_string_list,
        retweet_rate_rank_string_list = retweet_rate_rank_string_list,
        votes_present_string_list = votes_present_string_list,
        top_contrib_topic_list_list = top_contrib_topic_list_list,
        full_mla_list = sorted(mla_ids.normal_name.tolist()),
        postcodes_list = sorted(postcodes_to_constits.Postcode.tolist() + mla_ids.ConstituencyName.unique().tolist()),
        combined_demog_table_list = [e[1].values.tolist() for e in combined_demog_table[['constit','total_population',
            'mean_age','median_wage','pct_brought_up_protestant']].iterrows()],
        combined_demog_table2_list = [e[1].values.tolist() for e in combined_demog_table[['constit','total_area_sqkm',
            'pct_urban','n_farms','pct_work_in_agri','pct_adults_IS_claimants','pct_children_in_IS_households']].iterrows()],
        constit_population = f"{demogs_for_constit.total_population:,}",
        constit_second_message = comb_rank_text,
        constit_MDM_rank_order = combined_demog_table['MDM_mean_rank'].rank().astype(int).iloc[demogs_row_is_constit],
        constit_alphabetical_rank_order = combined_demog_table['constit'].rank().astype(int).iloc[demogs_row_is_constit],
        elct_fcst_constit_party_stuff = elct_fcst_constit_party_stuff,
        cands_table_code = cands_table,
        elct_n_ensemble = elct_n_ensemble)

#Most minister answers bars
# @app.route('/data/plot_minister_answers_bars')
# def plot_minister_answers_bars_fn():
#     selection = alt.selection_single(on='mouseover', empty='all')
#     plot = alt.Chart(minister_answers).mark_bar()\
#         .add_selection(selection)\
#         .encode(#x='Minister_normal_name', 
#             y='Questions answered',
#             x=alt.Y('Minister_normal_name', sort='-y', axis = alt.Axis(title=None)),
#             color = alt.Color('Minister_party_name', 
#                 scale=alt.Scale(
#                     domain=party_colours[party_colours.party_name.isin(minister_answers.Minister_party_name)]['party_name'].tolist(), 
#                     range=party_colours[party_colours.party_name.isin(minister_answers.Minister_party_name)]['colour'].tolist()
#                     )),
#             #opacity = alt.condition(selection, alt.value(1), alt.value(0.3)),
#             tooltip = 'tooltip_text')\
#         .properties(title = ' ', width='container', height=250)
#     #plot1 = plot1.configure_view(discreteWidth=800, continuousHeight=500)
#     #plot = plot.configure_title(fontSize=16, font='Courier')
#     plot = plot.configure_legend(disable=True)
 
#     return plot.to_json()

#
@app.route('/data/plot_minister_answer_times_<session_to_plot>')
def plot_minister_answer_times_fn(session_to_plot):
    #Problem with sorting two layer chart in Vega
    #Fixed in Altair 3 according to here https://github.com/altair-viz/altair/issues/820
    #  but seems not to be in my case
    #Sorting values first also seems not to work
    #Another workaround is to do second layer with independent axis, and hide labels and ticks,
    #  and this does work (https://github.com/altair-viz/altair/issues/820#issuecomment-386856394)

    if session_to_plot == CURRENT_ASSEMBLY_SESSION:
        plot_df = minister_time_to_answer
    else:
        plot_df = hist_minister_time_to_answer[hist_minister_time_to_answer.session_name==session_to_plot]

    plot = alt.Chart(plot_df).mark_bar(size=3)\
        .encode(x=alt.X('Questions answered', axis=alt.Axis(grid=True)),
            y=alt.Y('Minister_normal_name', sort=alt.EncodingSortField(order='ascending', field='Questions answered'),
                axis = alt.Axis(title=None)),
            color = alt.Color('Minister_party_name', 
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(plot_df.Minister_party_name)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(plot_df.Minister_party_name)]['colour'].tolist()
                    )))#\
        #.properties(title = ' ', width='container', height=250)
    
    # #Lose the axis ordering if add this on top
    #plot_b = plot.mark_circle(size=80)#\
        #.encode(x='Median days to answer',
        #    y=alt.Y('Minister_normal_name', sort=alt.EncodingSortField(order='ascending', field='tmp_sort_field')))
            #y=alt.Y('Minister_normal_name', sort='x'),
    #plot = plot + plot_b

    #default opacity is < 1 for circles so have to set to 1 to match bars
    plot_b = alt.Chart(plot_df).mark_circle(size=200, opacity=1)\
        .encode(x='Questions answered',
            y=alt.Y('Minister_normal_name', sort=alt.SortField(order='ascending', field='Questions answered'),
                axis = alt.Axis(labels=False, ticks=False, title=None)),
            color = alt.Color('Minister_party_name', 
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(plot_df.Minister_party_name)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(plot_df.Minister_party_name)]['colour'].tolist()
                    )),
            size = 'Median days to answer',
            tooltip = 'tooltip_text')#\
        #.properties(title = '', width=300, height=250)
    plot = alt.layer(plot, plot_b, data=plot_df).resolve_scale(y='independent')
    plot = plot.properties(width = 'container', 
        height = 300,
        background = 'none')

    plot = plot.configure_legend(disable=True)

    return plot.to_json()

#Most questions asked, split by written/oral
@app.route('/data/plot_questions_asked_<session_to_plot>_web')
def plot_questions_asked_fn_web(session_to_plot):
    return plot_questions_asked_fn(session_to_plot)

@app.route('/data/plot_questions_asked_<session_to_plot>_mobile')
def plot_questions_asked_fn_mobile(session_to_plot):
    return plot_questions_asked_fn(session_to_plot, mobile_mode = True)

def plot_questions_asked_fn(session_to_plot, mobile_mode = False):

    if session_to_plot == CURRENT_ASSEMBLY_SESSION:
        plot_df = questioners.sort_values('Questions asked', ascending=False)\
            .groupby('Question type').head(8 if mobile_mode else 12)
    else:
        plot_df = hist_questioners[hist_questioners.session_name == session_to_plot]\
            .sort_values('Questions asked', ascending=False)\
            .groupby('Question type').head(8 if mobile_mode else 12)

    #Opacity change on selection doesn't add anything
    #selection = alt.selection_single(on='mouseover', empty='all')
    #.add_selection(selection)\
    #opacity = alt.condition(selection, alt.value(1), alt.value(0.4)),

    plot = alt.Chart(plot_df).mark_bar(opacity=1)\
        .encode(
            y=alt.Y('Questions asked'),
            x=alt.X('normal_name', sort='-y', title=None),
            color = alt.Color('PartyName', 
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(plot_df.PartyName)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(plot_df.PartyName)]['colour'].tolist()
                    ), legend=None),
            facet = alt.Facet('Question type:N', columns=1),
            tooltip='tooltip_text:N')\
        .resolve_scale(x='independent', y = 'independent')\
        .properties(title = '', 
            width = 220 if mobile_mode else 440, 
            height = 250,
            background = 'none')
    #plot = plot.configure_title(fontSize=20, font='Courier')
    #plot = add_grey_legend(plot)
    # plot = plot.configure_legend(
    #     direction='horizontal', 
    #     orient='top',
    #     strokeColor='gray',
    #     fillColor='#EEEEEE',
    #     padding=10,
    #     cornerRadius=10
    # )
    #plot = plot.configure_legend(disable=True)

    return plot.to_json()

#Tweet volume and retweets scatter of parties
# @app.route('/data/plot_party_tweets_scatter')
# def plot_party_tweets_scatter_fn():

#     plot = alt.Chart(retweet_rate_last_month).mark_circle(size=140, opacity=1)\
#         .encode(x=alt.X('n_original_tweets', title='Number of original tweets'),
#             y=alt.Y('retweets_per_tweet', title='Retweets per tweet'),
#             color = alt.Color('mla_party', 
#                 scale=alt.Scale(
#                     domain=party_colours[party_colours.party_name.isin(retweet_rate_last_month.mla_party)]['party_name'].tolist(), 
#                     range=party_colours[party_colours.party_name.isin(retweet_rate_last_month.mla_party)]['colour'].tolist()
#                     )),
#                  #legend=alt.Legend(title="Party")),
#             tooltip = 'tooltip_text:N')\
#         .properties(title = '', width=500, height=500)        

#     #start fresh layer, otherwise fontSize is locked to size of points
#     text = alt.Chart(retweet_rate_last_month).mark_text(
#         align='left',
#         baseline='middle',
#         dx=6, dy=-6,
#         fontSize=11
#     ).encode(
#         text='mla_party', x='n_original_tweets', y='retweets_per_tweet'
#     )
#     plot += text

#     plot = plot.configure_legend(disable=True)
#     plot = plot.configure_title(fontSize=20, font='Courier')
#     #plot = plot.configure_axis(labelFontSize=14)
#     #plot = plot.configure_axisX(tickCount = x_ticks)

#     return plot.to_json()

#Most tweets by person
@app.route('/data/plot_user_tweetnum_web')
def plot_user_tweetnum_fn_web():
    return plot_user_tweetnum_fn()
@app.route('/data/plot_user_tweetnum_mobile')
def plot_user_tweetnum_fn_mobile():
    return plot_user_tweetnum_fn(mobile_mode = True)

def plot_user_tweetnum_fn(mobile_mode = False):
    top_n_tweeters = tweets_df[tweets_df.created_at_yweek.isin(last_5_yweeks_tweets)].groupby('normal_name').status_id.count()\
        .reset_index().sort_values('status_id')\
        .tail(10 if mobile_mode else 15)\
        .normal_name.tolist()
    top_tweeters_plot_df = top_tweeters[top_tweeters.normal_name.isin(top_n_tweeters)]

    selection = alt.selection_single(on='mouseover', empty='all')
    plot = alt.Chart(top_tweeters_plot_df).mark_bar()\
        .add_selection(selection)\
        .encode(
            y=alt.Y('n_tweets', title='Number of tweets'),
            x=alt.Y('normal_name', sort='-y', title=None),
            color = alt.Color('tweet_type', 
                scale=alt.Scale(
                    domain=['original','retweet'],
                    range=['Peru','SlateGrey']
                ), legend=alt.Legend(title="")),
            opacity = alt.condition(selection, alt.value(1), alt.value(0.3)),
            tooltip='tooltip_text:N')\
        .properties(title = '', 
            width = 'container', 
            height = 200 if mobile_mode else 300, 
            background='none')

    plot = add_grey_legend(plot, mobile_mode = mobile_mode)

    plot = plot.to_json()

    return plot

#Highest retweet rates - need to take .head(15) here
@app.route('/data/plot_user_retweet_web')
def plot_user_retweet_fn_web():
    return plot_user_retweet_fn()

@app.route('/data/plot_user_retweet_mobile')
def plot_user_retweet_fn_mobile():
    return plot_user_retweet_fn(mobile_mode = True)

def plot_user_retweet_fn(mobile_mode = False):
    plot_df = member_retweets.head(10 if mobile_mode else 15)
    selection = alt.selection_single(on='mouseover', empty='all')
    plot = alt.Chart(plot_df).mark_bar()\
        .add_selection(selection)\
        .encode(
            y=alt.Y('retweets_per_tweet', title='Retweets per tweet'),
            x=alt.Y('normal_name', sort='-y', title=None),
            color = alt.Color('mla_party', 
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(plot_df.mla_party)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(plot_df.mla_party)]['colour'].tolist()
                    ), legend=alt.Legend(title='')),
            tooltip='tooltip_text:N')\
        .properties(title = '', 
            width = 'container', 
            height = 200 if mobile_mode else 300,
            background = 'none')

    plot = add_grey_legend(plot, mobile_mode = mobile_mode)

    return plot.to_json()

#Highest and lowest tweet sentiment scores
@app.route('/data/plot_user_tweet_sentiment_web')
def plot_user_tweet_sentiment_fn_web():
    return plot_user_tweet_sentiment_fn()

@app.route('/data/plot_user_tweet_sentiment_mobile')
def plot_user_tweet_sentiment_fn_mobile():
    return plot_user_tweet_sentiment_fn(mobile_mode = True)

def plot_user_tweet_sentiment_fn(mobile_mode = False):
    n_of_each_to_plot = 6 if mobile_mode else 10
    df_to_plot = member_tweet_sentiment[member_tweet_sentiment.normal_name.isin(
        member_tweet_sentiment.sort_values('sentiment_vader_compound').head(n_of_each_to_plot).normal_name.tolist() +
        member_tweet_sentiment.sort_values('sentiment_vader_compound').tail(n_of_each_to_plot).normal_name.tolist()
    )].sort_values('sentiment_vader_compound')
    df_to_plot['rank_group'] = ['highest']*n_of_each_to_plot + ['lowest']*n_of_each_to_plot

    #overall_av_sentiment_score = member_tweet_sentiment.sentiment_vader_compound.mean()
    #df_to_plot['sentiment_vader_compound'] = df_to_plot['sentiment_vader_compound'] - overall_av_sentiment_score    

    #most_pos_text_y2 = df_to_plot.sentiment_vader_compound.min()*0.22 #some way below the zero line relative to the lowest point drawn
    #most_neg_text_y2 = df_to_plot.sentiment_vader_compound.iloc[int(0.5*n_of_each_to_plot)]+0.04
    most_pos_text_y2 = df_to_plot.sentiment_vader_compound.min()*0.15
    most_neg_text_y2 = df_to_plot.sentiment_vader_compound.max()*0.15

    df_to_plot['y2'] = [most_neg_text_y2]*n_of_each_to_plot + [most_pos_text_y2]*n_of_each_to_plot
    df_to_plot['text'] = ['']*df_to_plot.shape[0]
    df_to_plot.iloc[int(n_of_each_to_plot/2-1), df_to_plot.columns.get_loc('text')] = 'Most negative'
    df_to_plot.iloc[df_to_plot.shape[0]-int(n_of_each_to_plot/2), df_to_plot.columns.get_loc('text')] = 'Most positive'
    df_to_plot['names_numbered'] = range(df_to_plot.shape[0])

    base = alt.Chart(df_to_plot)\
        .encode(
            y=alt.Y('sentiment_vader_compound', title='Mean tweet sentiment score'),
            x=alt.X('normal_name', sort=alt.EncodingSortField(field='sentiment_vader_compound', op='max'), axis = alt.Axis(title=None)),
            color = alt.Color('mla_party',
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(df_to_plot.mla_party)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(df_to_plot.mla_party)]['colour'].tolist()
                    ),
                legend = alt.Legend(title = '')),
            tooltip='tooltip_text:N')
    
    plot = base.mark_bar(size=3) + base.mark_circle(size=200, opacity=1)

    dividing_line = alt.Chart(pd.DataFrame({'anything': [0]})).mark_rule(yOffset=0, strokeDash=[2,2])\
       .encode(y=alt.Y('anything'))
    plot += dividing_line

    text = alt.Chart(df_to_plot).mark_text(
        align='center',
        baseline='middle',
        dx=0, 
        dy=0,
        fontSize = 13 if mobile_mode else 15
    ).encode(text='text', y='y2',
        x=alt.X('normal_name', sort=alt.EncodingSortField(field='sentiment_vader_compound', op='max')), 
    ).transform_filter(alt.datum.text != '')
    plot += text

    plot = plot.properties(width = 'container', 
        height = 250 if mobile_mode else 400,
        background = 'none')

    plot = add_grey_legend(plot, orient = 'bottom-right', mobile_mode = mobile_mode)

    return plot.to_json()

#Tweets PCA scatter of politicians}
@app.route('/data/plot_tweet_pca_all_mlas_web')
def plot_tweet_pca_all_mlas_fn_web():
    return plot_tweet_pca_all_mlas_fn()

@app.route('/data/plot_tweet_pca_all_mlas_mobile')
def plot_tweet_pca_all_mlas_fn_mobile():
    return plot_tweet_pca_all_mlas_fn(mobile_mode = True)

def plot_tweet_pca_all_mlas_fn(mobile_mode = False):

    plot = alt.Chart(tweet_pca_positions).mark_circle(opacity=0.6)\
        .encode(x=alt.X('mean_PC1', 
                    #axis=alt.Axis(title='Principal component 1 (explains 12% variance)', labels=False)),
                    axis=alt.Axis(title=['<---- more asking for good governance','             more historical references or Irish language ---->'], labels=False)),
            y=alt.Y('mean_PC2', #axis=alt.Axis(title='Principal component 2 (explains 2% variance)', labels=False)),
                axis=alt.Axis(title='more praising others <----    ----> more brexit and party politics', labels=False)),
            color = alt.Color('mla_party', 
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(tweet_pca_positions.mla_party)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(tweet_pca_positions.mla_party)]['colour'].tolist()
                    ),
                legend=alt.Legend(title='')),
            size = alt.Size('num_tweets', scale=alt.Scale(range=[30 if mobile_mode else 60, 250 if mobile_mode else 500]),
                legend=None),
            tooltip = 'tooltip_text:N')\
        .properties(title = '', 
            width = 'container', 
            height = 250 if mobile_mode else 450,  #stretch x-axis because PC1 explains more variance?
            background = 'none')

    if not mobile_mode:
        plot = plot.encode(href = 'indiv_page_url:N')

    plot = add_grey_legend(plot, orient = 'top', columns = 2 if mobile_mode else 4,
        mobile_mode = mobile_mode)
    #plot = plot.configure_axis(titleFontSize=10)

    return plot.to_json()

#Top news sources overall
@app.route('/data/plot_news_sources_web')
def plot_news_sources_fn_web():
    return plot_news_sources_fn()

@app.route('/data/plot_news_sources_mobile')
def plot_news_sources_fn_mobile():
    return plot_news_sources_fn(mobile_mode = True)

def plot_news_sources_fn(mobile_mode = False):
    topN_sources = news_sources.groupby('source').agg({'News articles': sum})\
        .reset_index().sort_values('News articles')\
        .tail(10 if mobile_mode else 15)\
        .source.tolist()
    news_sources_plot_df = news_sources[news_sources.source.isin(topN_sources)]

    selection = alt.selection_single(on='mouseover', empty='all')
    plot = alt.Chart(news_sources_plot_df).mark_bar()\
        .add_selection(selection)\
        .encode(
            y=alt.Y('News articles'),
            x=alt.X('source', sort='-y', title=None),
            color = alt.Color('PartyGroup', 
                scale=alt.Scale(
                    domain=['Unionist','Other','Nationalist'],
                    range=['RoyalBlue','Moccasin','LimeGreen']
                ), legend = alt.Legend(title = 'Mentioning')),
            opacity = alt.condition(selection, alt.value(1), alt.value(0.4)),
            tooltip='tooltip_text:N')\
        .configure_axis(labelAngle=-45)\
        .properties(title = '', 
            width = 'container', 
            height = 200 if mobile_mode else 300,
            background = 'none')
    plot = add_grey_legend(plot, mobile_mode = mobile_mode)

    return plot.to_json() 

#News sentiment by party and week
@app.route('/data/plot_news_volume_web')
def plot_news_volume_fn_web():
    return shared_plot_news_fn(news_sentiment_by_party_week, 'vol_smooth', 'Number of mentions', '')

@app.route('/data/plot_news_volume_mobile')
def plot_news_volume_fn_mobile():
    return shared_plot_news_fn(news_sentiment_by_party_week, 'vol_smooth', 'Number of mentions', '', mobile_mode=True)

def shared_plot_news_fn(news_sentiment_by_party_week, y_variable, y_title, title, mobile_mode=False):
    #x_ticks = news_sentiment_by_party_week.published_date_week.nunique()
    #x_range = (news_sentiment_by_party_week.published_date_week.min(),
    #    news_sentiment_by_party_week.published_date_week.max())
    news_sentiment_by_party_week['plot_date'] = news_sentiment_by_party_week.apply(
        lambda row: pd.to_datetime('{}-01-01'.format(row['published_date_year'])) + pd.to_timedelta(7*(row['published_date_week']-1), unit='days'),
        axis=1)

    plot = alt.Chart(news_sentiment_by_party_week).mark_line(size=5)\
        .encode(x=alt.X('plot_date:T', title=None),
            y=alt.Y(y_variable, axis=alt.Axis(title=y_title, minExtent=10, maxExtent=100)),
            color = alt.Color('PartyName', 
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(news_sentiment_by_party_week.PartyName)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(news_sentiment_by_party_week.PartyName)]['colour'].tolist()
                    ),
                legend=alt.Legend(title='')))\
        .configure_axis(labelAngle=-30)\
        .properties(title = title,
            width = 'container', 
            height = 200 if mobile_mode else 300,
            background = 'none')

    if not mobile_mode:
        plot = plot.configure_axis(labelFontSize = 14)

    plot = plot.configure_legend(
        direction='horizontal', 
        orient='top',
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding = 7 if mobile_mode else 10,
        cornerRadius = 10,
        columns = 2 if mobile_mode else 5
    )

    return plot.to_json()

@app.route('/data/plot_news_sentiment')
def plot_news_sentiment_fn(mobile_mode = False):
    #There seems to be a bug in Vega boxplot continuous title, possibly 
    #  related to it being a layered plot - gives title of 'variable, mytitle, mytitle'
    #title=None gives no title; title='' gives title=variable name
    #news_sentiment_by_party_week['Sentiment score'] = news_sentiment_by_party_week.sr_sentiment_score
    #work out order manually - easier than figuring out the altair method
    party_order = news_sentiment_by_party_week.groupby('PartyName').sr_sentiment_score.mean().sort_values(ascending=False).index.tolist()

    plot = alt.Chart(news_sentiment_by_party_week).mark_boxplot(size=30)\
        .encode(y = alt.Y('PartyName:N', title=None, sort=party_order),
            #x = alt.X('Sentiment score:Q', title=''),
            x = alt.X('sr_sentiment_score', title=None),
            color = alt.Color('PartyName', 
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(news_sentiment_by_party_week.PartyName)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(news_sentiment_by_party_week.PartyName)]['colour'].tolist()
                    ),
                legend=None),
            tooltip=alt.Tooltip(field='tooltip_text', type='nominal', aggregate='max'))\
        .properties(title = '',
            width='container', 
            height = 200 if mobile_mode else 300,
            background = 'none')
    #use the aggregated tooltip to avoid printing the full summary
    #  with long floats and variable names

    plot = plot.configure_legend(
        direction='horizontal', 
        orient='top',
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=10
    )

    return plot.to_json()


#Polls tracker
@app.route('/data/polls_plot_web')
def polls_plot_fn_web():
    return polls_plot_fn()
@app.route('/data/polls_plot_mobile')
def polls_plot_fn_mobile():
    return polls_plot_fn(mobile_mode = True)

def polls_plot_fn(mobile_mode = False):
    #x_ticks = polls_df.discoverDate_week.nunique()
    #x_range = (polls_df.date.min() - pd.to_timedelta('90 days'),
    #    polls_df.date.max() + pd.to_timedelta('90 days'))
    #selection = alt.selection_single(on='mouseover', empty='all')
    selection2 = alt.selection_single(fields=['pct_type'], bind='legend')

    polls_df['pct_type'] = 'polls'
    elections_df['pct_type'] = 'elections'
    joint_plot_df = pd.concat([
        polls_df[['date','pct','party','tooltip_text','pct_type']],
        elections_df[['date','pct','party','tooltip_text','pct_type']]
    ])
    #Alternative to the scroll option is to only show last 3 years on mobile
    if False and mobile_mode:
        years_to_show_on_mobile = 3
        joint_plot_df = joint_plot_df[joint_plot_df.date >= (datetime.datetime.today() - datetime.timedelta(days=years_to_show_on_mobile*365))]
        poll_avgs_track_plot_df = poll_avgs_track[poll_avgs_track.date >= (datetime.datetime.today() - datetime.timedelta(days=years_to_show_on_mobile*365))]
    else:
        poll_avgs_track_plot_df = poll_avgs_track

    plot = alt.Chart(joint_plot_df).mark_point(filled=True, size=100 if mobile_mode else 150)\
        .encode(x=alt.X('date:T', 
                    axis=alt.Axis(title='', tickCount = joint_plot_df.date.dt.year.nunique()),
                    scale=alt.Scale(domain=((datetime.date.today() - datetime.timedelta(days=4*365)).strftime('%Y-%m-%d'), poll_avgs_track_plot_df['date'].max()))),
            y=alt.Y('pct', axis=alt.Axis(title='Vote share / %'), scale=alt.Scale(domain=(0,50))),
            color = alt.Color('party', 
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(polls_df.party)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(polls_df.party)]['colour'].tolist()
            )),
            shape = alt.Shape('pct_type', legend=alt.Legend(title='show...'),
                scale=alt.Scale(domain = ['polls','elections'], range = ['circle','diamond'])),
            size = alt.condition(alt.datum.pct_type == 'polls', alt.value(150), alt.value(300)),
            opacity = alt.condition(selection2, alt.value(0.5), alt.value(0.1)),
            tooltip = 'tooltip_text:N')\
        .interactive(bind_x=True, bind_y=False)\
        .add_selection(selection2)

    #Draw lines with separate data to get control on size
    plot2 = alt.Chart(poll_avgs_track_plot_df).mark_line(size=3)\
        .encode(x='date:T', y='pred_pct',
            color = alt.Color('party',
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(poll_avgs_track_plot_df.party)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(poll_avgs_track_plot_df.party)]['colour'].tolist()
                    ), 
                legend = alt.Legend(title = ''))
            )

    #plot = alt.layer(plot, plot2, selectors, points, text, rules, plot3)
    plot = alt.layer(plot2, plot)#, plot3)
    # plot = plot.configure_title(fontSize=20, font='Courier')
    plot = plot.configure_axis(titleFontSize=14, labelFontSize=10)
    plot = plot.configure_legend(
        direction='horizontal', 
        orient='top-right',
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=10,
        columns = 4,
        #columns = 2 if mobile_mode else 4,  #if not using scroll, use this
        labelFontSize = 8 if mobile_mode else 12,
        symbolSize = 40 if mobile_mode else 60
    )

    plot = plot.properties(title = '', 
        width = 600 if mobile_mode else 'container',  #rely on horizontal scroll on mobile
        height = 400,
        background = 'none')

    return plot.to_json()

#Votes PCA plot of all MLAs 
@app.route('/data/plot_vote_pca_all_mlas_<session_to_plot>_web')
def plot_vote_pca_all_mlas_fn_web(session_to_plot):
    return plot_vote_pca_all_mlas_fn(session_to_plot)

@app.route('/data/plot_vote_pca_all_mlas_<session_to_plot>_mobile')
def plot_vote_pca_all_mlas_fn_mobile(session_to_plot):
    return plot_vote_pca_all_mlas_fn(session_to_plot, mobile_mode=True)

def plot_vote_pca_all_mlas_fn(session_to_plot, mobile_mode = False):
    if session_to_plot == CURRENT_ASSEMBLY_SESSION:
        plot_df = mlas_2d_rep
        pct_explained = (100*my_pca.explained_variance_ratio_[0], 100*my_pca.explained_variance_ratio_[1])
    else:
        plot_df = hist_votes_pca_res[session_to_plot][0]
        pct_explained = hist_votes_pca_res[session_to_plot][1]

    plot = alt.Chart(plot_df).mark_circle(size = 60 if mobile_mode else 120, opacity=0.6)\
        .encode(x=alt.X('x', 
                    axis=alt.Axis(title=['Principal component 1','(explains {:.0f}% variance)'.format(pct_explained[0])], labels=False)),
            y=alt.Y('y', axis=alt.Axis(title='Principal component 2 (explains {:.0f}% variance)'.format(pct_explained[1]), labels=False)),
            color = alt.Color('party', 
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(plot_df.party)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(plot_df.party)]['colour'].tolist()
                    ),
                 legend=alt.Legend(title="")),
            #href = 'indiv_page_url:N',
            tooltip = 'normal_name:N')\
        .properties(title = '', 
            width = 'container', 
            height = 250 if mobile_mode else 500,
            background = 'none')
        #.configure_axisX(tickCount = 0)\
        #.configure_axisY(tickCount = 0)

    if session_to_plot == CURRENT_ASSEMBLY_SESSION:
        plot = plot.encode(href = 'indiv_page_url:N')

    plot = plot.configure_legend(
        direction='horizontal', 
        orient='top',
        strokeColor='gray',
        fillColor='#EEEEEE',
        padding=10,
        cornerRadius=10,
        columns=4,
        labelFontSize = 8 if mobile_mode else 12,
        symbolSize = 40 if mobile_mode else 60)
    #plot = plot.configure_legend(disable=True)

    #start fresh layer, otherwise fontSize is locked to size of points
    # text = alt.Chart(mlas_2d_rep[mlas_2d_rep.normal_name==mla_choice]).mark_text(
    #     align='left',
    #     baseline='middle',
    #     dx=12,
    #     fontSize=15
    # ).encode(
    #     text='normal_name', x='x', y='y'
    # )
    # plot += text

    return plot.to_json()

#How often party votes as one
# @app.route('/data/plot_party_unity_bars_<session_to_plot>')
# def plot_party_unity_bars_fn(session_to_plot):

#     if session_to_plot == CURRENT_ASSEMBLY_SESSION:
#         plot_df = votes_party_unity
#     else:
#         plot_df = hist_votes_party_unity[hist_votes_party_unity.session_name==session_to_plot]

#     selection = alt.selection_single(on='mouseover', empty='all')
#     plot = alt.Chart(plot_df).mark_bar()\
#         .add_selection(selection)\
#         .encode(y = 'Percent voting as one',
#             x = alt.Y('PartyName', sort='-y', title=None),
#             color = alt.Color('PartyName', 
#                 scale=alt.Scale(
#                     domain=party_colours[party_colours.party_name.isin(plot_df.PartyName)]['party_name'].tolist(), 
#                     range=party_colours[party_colours.party_name.isin(plot_df.PartyName)]['colour'].tolist()
#                     )),
#             tooltip = 'tooltip_text')\
#         .properties(title = ' ', 
#             width = 'container', 
#             height = 300,
#             background = 'none')
#     plot = plot.configure_legend(disable=True)
 
#     return plot.to_json()

#Top plenary topics
@app.route('/data/plot_plenary_topics_overall_<session_to_plot>')
def plot_plenary_topics_overall_fn(session_to_plot):

    if session_to_plot == CURRENT_ASSEMBLY_SESSION:
        plot_df = plenary_contribs_topic_counts
    else:
        plot_df = hist_plenary_contribs_topic_counts[hist_plenary_contribs_topic_counts.session_name==session_to_plot]

    topic_names = list(plenary_contribs_colour_dict.keys())

    selection = alt.selection_single(on='mouseover', empty='all')
    plot = alt.Chart(plot_df).mark_bar(size=25)\
        .add_selection(selection)\
        .encode(
            x=alt.X('n_contribs', title='Number of instances'),
            y=alt.Y('topic_name', sort='-x', title=None),
            color=alt.Color('topic_name', 
                scale=alt.Scale(
                    domain=topic_names, 
                    range=[plenary_contribs_colour_dict[t] for t in topic_names]
                )
            ),
            tooltip='tooltip_text:N')\
        .properties(title = ' ', 
            width = 'container', 
            height = 450,
            background = 'none')\
        .configure_legend(disable=True)
    plot = plot.configure_axisY(labelFontSize=9)

    return plot.to_json()

#Plenary emotion scores
@app.route('/data/plot_plenary_emotions_by_party_<session_to_plot>_web')
def plot_plenary_emotions_fn_web(session_to_plot):
    return plot_plenary_emotions_fn(session_to_plot)

@app.route('/data/plot_plenary_emotions_by_party_<session_to_plot>_mobile')
def plot_plenary_emotions_fn_mobile(session_to_plot):
    return plot_plenary_emotions_fn(session_to_plot, mobile_mode=True)

def plot_plenary_emotions_fn(session_to_plot, mobile_mode = False):
    selection_by_party = alt.selection_single(on='mouseover', empty='all', encodings=['color'])

    #TODO would be nice to have on hover, selected party vs average of the rest

    #fear and disgust are strongly correlated with anger and sadness, so can omit
    #surprise values are very similar for the 5 parties
    if session_to_plot == CURRENT_ASSEMBLY_SESSION:
        emotions_party_to_plot = emotions_party_agg[(emotions_party_agg.emotion_type.isin(
            ['anger','anticipation','joy','sadness','trust']
            )) & (emotions_party_agg.PartyName.isin(['Alliance','DUP','SDLP','Sinn Fein','UUP']))]
    else:
        emotions_party_to_plot = hist_emotions_party_agg[(hist_emotions_party_agg.session_name==session_to_plot) & 
            (hist_emotions_party_agg.emotion_type.isin(['anger','anticipation','joy','sadness','trust'])) &
            (hist_emotions_party_agg.PartyName.isin(['Alliance','DUP','SDLP','Sinn Fein','UUP']))]

    emotions_party_to_plot = emotions_party_to_plot.merge(
        pd.DataFrame({'order': [1,2,3,4,5], 'emotion_type': ['trust','anticipation','joy','sadness','anger']}),
        on = 'emotion_type', how = 'inner'
    )

    plot = alt.Chart(emotions_party_to_plot).mark_point(size=120 if mobile_mode else 200, strokeWidth=5 if mobile_mode else 6)\
        .add_selection(selection_by_party)\
        .encode(
            x = alt.X('ave_emotion', title='Fraction words scoring'),
            y = alt.Y('emotion_type', title=None,
                sort=alt.EncodingSortField(order='ascending', field='order')),
            color = alt.Color('PartyName', 
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(emotions_party_to_plot.PartyName)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(emotions_party_to_plot.PartyName)]['colour'].tolist()
                ), legend=None),
            opacity = alt.condition(selection_by_party, alt.value(0.7), alt.value(0.05)),
            tooltip = 'PartyName'
        )\
        .properties(height = 200 if mobile_mode else 250, 
            width = 'container',
            background = 'none')

    return plot.to_json()


#Constituency metrics on the postcode page
@app.route('/data/plot_constituency_depriv_metrics_<constit_choice>_web')
def plot_constituency_depriv_metrics_fn_web(constit_choice):
    return plot_constituency_depriv_metrics_fn(constit_choice)

@app.route('/data/plot_constituency_depriv_metrics_<constit_choice>_mobile')
def plot_constituency_depriv_metrics_fn_mobile(constit_choice):
    return plot_constituency_depriv_metrics_fn(constit_choice, mobile_mode = True)

def plot_constituency_depriv_metrics_fn(constit_choice, mobile_mode = False):
    source = combined_demog_table.melt(id_vars='constit', 
        value_vars=[c for c in combined_demog_table.columns if c[-9:]=='mean_rank'],
        value_name='score', var_name='metric')

    depriv_col_pretty_names = {
        'MDM_mean_rank': 'Mean Deprivation Measure',
        'proximity_to_services_depriv_mean_rank': 'Proximity to services',
        'living_environment_depriv_mean_rank': 'Living environment',
        'crime_disorder_depriv_mean_rank': 'Crime & disorder',
        'income_depriv_mean_rank': 'Income',
        'employment_depriv_mean_rank': 'Employment'
    }
    source['metric'] = source.metric.apply(lambda c: depriv_col_pretty_names[c])

    source['highlighted'] = [1 if c.upper()==constit_choice else 0 for c in source.constit.tolist()]
    source['tooltip_text'] = source.apply(lambda row: f"{row['constit']}: {row['score']} for deprivation in {row['metric'].lower()}".replace('mean deprivation measure','MDM'), axis=1)

    selection = alt.selection_single(on='mouseover', fields=['constit'])

    plot = alt.Chart(source).mark_circle(stroke='black', strokeWidth=0).encode(
        alt.X('score:Q', scale=alt.Scale(domain=[780,80]), axis=alt.Axis(title='Deprivation mean ward rank', values=[100,300,500,700])),
        alt.Y('metric:N', axis=alt.Axis(title=None), scale=alt.Scale(domain=['Mean Deprivation Measure','Crime & disorder',
                                                                             'Employment','Income',
                                                                             'Living environment','Proximity to services'])),
        size = alt.Size('highlighted:Q', scale=alt.Scale(range=[100,600], domain=[0,1])),
        opacity = alt.condition(selection, alt.value(1), alt.value(0.2)),
        color = alt.Color('highlighted:Q', scale=alt.Scale(range=['LightSteelBlue','#7a527a'], domain=[0,1]), legend=None),
        tooltip='tooltip_text:N'
    ).add_selection(selection)

    annotations = alt.Chart(pd.DataFrame({'y': [9.6,9.6,10], 'x': [150,780,0], 
            'text': ['most deprived','least deprived','']})).mark_text(
        align = 'left', baseline = 'middle',
        fontSize = 15, color = 'grey', dx = 7
    ).encode(
        x='x',
        y=alt.Y('y', axis=None),
        text='text'
    )

    plot = plot + annotations
    
    plot = plot.properties(
        #title=f'Deprivation scores for {constit_choice} vs other constituencies',
        title='',
        height = 250 if mobile_mode else 350, 
        width = 'container',
        background = 'none'
    )

    plot = plot.configure_axis(titleFontSize=14, labelFontSize=13, grid=False)
    plot = plot.configure_title(fontSize=18, font='Arial')

    return plot.to_json()

#Election cw pct bars
#TODO change file to group Ind, small parties together
@app.route('/data/elct_cw_bars_plot_web')
def elct_cw_bars_fn_web():
    return elct_cw_bars_fn()
@app.route('/data/elct_cw_bars_plot_mobile')
def elct_cw_bars_fn_mobile():
    return elct_cw_bars_fn(mobile_mode = True)

def elct_cw_bars_fn(mobile_mode = False):
    elct_fcst_cw_fps['tooltip_text'] = elct_fcst_cw_fps.apply(
        lambda row: f"{row['party_short']} 2022 predicted {row['cw_pct']:.1f}%" if row['year']==2022 else f"{row['party_short']} 2017 first pref. {row['cw_pct']:.1f}%",
        axis=1
    )

    plot = alt.Chart(elct_fcst_cw_fps).mark_bar(width = 20 if mobile_mode else 32)\
        .encode(
            y=alt.Y('cw_pct', title='First pref. vote / %', axis=alt.Axis(grid=False, 
                labelFontSize = 10 if mobile_mode else 13, 
                titleFontSize = 11 if mobile_mode else 14, 
                labelPadding = 3 if mobile_mode else 5, 
                titlePadding = 5 if mobile_mode else 18)),
            x=alt.X('year:N', sort='ascending', axis=alt.Axis(grid=False, labels=False, ticks=False, title=None)),
            column=alt.Column('party_short:N', sort=alt.EncodingSortField(field='y', op='max'),
                spacing = 4 if mobile_mode else 15,
                title=None, 
                header=alt.Header(orient='bottom', 
                    labelFontSize = 10 if mobile_mode else 13, 
                    labelAngle = -30 if mobile_mode else 0, 
                    labelPadding = 280 if mobile_mode else 275,
                    labelAlign='center', labelOrient='bottom')),
            opacity=alt.Opacity('year', scale=alt.Scale(range=[0.4,1]), legend=None),
            color = alt.Color('party_short', 
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(elct_fcst_cw_fps.party_short)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(elct_fcst_cw_fps.party_short)]['colour'].tolist()
                    ), legend=None),
            tooltip='tooltip_text:N')\
        .properties(title = '', 
            #width here is for each Column, and has to fit with mark_bar(width) to give the right overlap
            width = 22 if mobile_mode else 48,
            height = 250,
            background = 'none')\
        .configure_view(stroke='transparent')

    return plot.to_json()


#Election cw seats bubble plot
@app.route('/data/elct_cw_seats_range_plot_web')
def elct_cw_seats_range_fn_web():
    return elct_cw_seats_range_fn()
@app.route('/data/elct_cw_seats_range_plot_mobile')
def elct_cw_seats_range_fn_mobile():
    return elct_cw_seats_range_fn(mobile_mode = True)

def elct_cw_seats_range_fn(mobile_mode = False):
    elct_n_ensemble = elct_fcst_ens_party_seats.it.max()    
    elct_seats_vcounts = elct_fcst_ens_party_seats[['party_short','n_seats']].value_counts().reset_index().rename(columns={0: 'n_cases'})
    elct_seats_vcounts['tooltip_text_individual'] = elct_seats_vcounts.apply(lambda row: f"{row['party_short']} win {row['n_seats']} seat{'s' if row['n_seats'] != 1 else ''} in {row['n_cases']}/{elct_n_ensemble} simulations", axis=1)

    elct_seats_vcounts['seat_cases'] = elct_seats_vcounts['n_seats'] * elct_seats_vcounts['n_cases']
    tmp = elct_seats_vcounts.groupby('party_short', as_index=False).agg(mean_n_seats = ('seat_cases', lambda x: sum(x)/elct_n_ensemble))
    tmp['tooltip_text_mean'] = tmp.apply(lambda row: f"{row['party_short']} are predicted to win an average of {row['mean_n_seats']:.1f} seats", axis=1)
    tmp = elct_seats_vcounts.merge(tmp, how='left', on='party_short')
    y_party_order = tmp.sort_values('mean_n_seats', ascending=False).party_short.unique().tolist()
    tmp['party_short'] = tmp.party_short.apply(lambda p: 'Ind.' if p=='Independent' else p)

    p1 = alt.Chart(tmp).mark_circle(
        opacity=0.7 #, stroke='black', strokeWidth=0.5
    ).encode(
        alt.X('n_seats:Q', axis=alt.Axis(labelFontSize = 10 if mobile_mode else 13, 
            titleFontSize = 11 if mobile_mode else 14, tickCount=8, grid=False, title='Number of seats')),
        alt.Y('party_short:N', 
            sort=y_party_order,
            title=None, axis=alt.Axis(labelFontSize = 11 if mobile_mode else 13, 
                labelPadding = 10 if mobile_mode else 20)),
        alt.Size('n_cases:Q',
            scale=alt.Scale(range=[20, 2000], type='pow', exponent=1.5),
            legend=None
        ),
        color = alt.Color('party_short', 
                    scale=alt.Scale(
                        domain=party_colours[party_colours.party_name.isin(tmp.party_short)]['party_name'].tolist(), 
                        range=party_colours[party_colours.party_name.isin(tmp.party_short)]['colour'].tolist()
                        ), 
                    legend=None),
        tooltip = 'tooltip_text_individual:N'
    )

    p2 = alt.Chart(tmp.drop_duplicates(subset=['party_short']))\
        .mark_tick(size=30, thickness=6, opacity=0.3, stroke='black')\
        .encode(
            x='mean_n_seats:Q',
            y=alt.Y('party_short:N', sort=y_party_order),
            color = alt.Color('party_short', 
                        scale=alt.Scale(
                            domain=party_colours[party_colours.party_name.isin(tmp.party_short)]['party_name'].tolist(), 
                            range=party_colours[party_colours.party_name.isin(tmp.party_short)]['colour'].tolist()
                            ), 
                        legend=None),
            tooltip = 'tooltip_text_mean:N'
        )

    plot = (p1 + p2).properties(
        #width = 220 if mobile_mode else 440,
        width = 'container',
        height = 320
    )

    return plot.to_json()

#Election party delta pcts faceted by constit
@app.route('/data/elct_cw_delta_seats_map_plot_web')
def elct_cw_delta_seats_map_fn_web():
    return elct_cw_delta_seats_map_fn()
@app.route('/data/elct_cw_delta_seats_map_plot_mobile')
def elct_cw_delta_seats_map_fn_mobile():
    return elct_cw_delta_seats_map_fn(mobile_mode = True)

def elct_cw_delta_seats_map_fn(mobile_mode = False):

    map = alt.Chart(alt.topo_feature('https://martinjc.github.io/UK-GeoJSON/json/ni/topo_wpc.json', feature='wpc')).mark_geoshape(
        fill='#fdfbee',
        stroke='gray'
    )

    constit_coords = pd.DataFrame([
        {'constit': 'East Antrim', 'lon': -5.87285, 'lat': 54.81905},
        {'constit': 'East Belfast', 'lon': -5.85854, 'lat': 54.60098},
        {'constit': 'East Londonderry', 'lon': -6.84, 'lat': 55.05},
        {'constit': 'Fermanagh and South Tyrone', 'lon': -7.588976, 'lat': 54.287027},
        {'constit': 'Foyle', 'lon': -7.30, 'lat': 54.98},
        {'constit': 'Lagan Valley', 'lon': -6.12, 'lat': 54.44},
        {'constit': 'Mid Ulster', 'lon': -6.72667, 'lat': 54.69},
        {'constit': 'Newry and Armagh', 'lon': -6.58295, 'lat': 54.243925},
        {'constit': 'North Antrim', 'lon': -6.25, 'lat': 54.94},
        {'constit': 'North Belfast', 'lon': -5.93428, 'lat': 54.68},
        {'constit': 'North Down', 'lon': -5.64764, 'lat': 54.65096},
        {'constit': 'South Antrim', 'lon': -6.25, 'lat': 54.68273},
        {'constit': 'South Belfast', 'lon': -5.92457, 'lat': 54.525},
        {'constit': 'South Down', 'lon': -6.05, 'lat': 54.20305},
        {'constit': 'Strangford', 'lon': -5.65, 'lat': 54.5},
        {'constit': 'Upper Bann', 'lon': -6.41, 'lat': 54.477043},
        {'constit': 'West Belfast', 'lon': -6.07, 'lat': 54.58},
        {'constit': 'West Tyrone', 'lon': -7.32, 'lat': 54.668427}
    ])

    tmp = elct_fcst_seat_deltas.merge(constit_coords, how='inner', on='constit')

    tmp['abs_prob_seat_change'] = tmp.apply(lambda row: max([row['prob_gain_seat'], row['prob_lose_seat']]), axis=1)
    tmp['dir_prob_seat_change'] = tmp.apply(lambda row: +1 if row['prob_gain_seat'] > 0 else -1, axis=1)
    tmp['tooltip_text'] = tmp.apply(lambda row: f"{row['party_short']} have a{'n' if row['abs_prob_seat_change']//0.1 == 8 or row['abs_prob_seat_change'] in [0.08,0.11,0.18] else ''} {row['abs_prob_seat_change']*100:.0f}% chance of {'gaining' if row['dir_prob_seat_change']==1 else 'losing'} a seat in {row['constit']}", axis=1)

    tmp['row_order'] = tmp.groupby(['constit','dir_prob_seat_change'])['abs_prob_seat_change'].rank(method='first', ascending=False) - 1
    tmp['lon_delta'] = tmp.row_order.apply(lambda x: (1 if x % 2 == 1 else -1) * ((x+1) // 2))
    tmp = tmp.merge(tmp.groupby('constit', as_index=False).agg(max_abs_change = ('abs_prob_seat_change',max)), how='inner', on='constit')
    tmp['lon_plot'] = tmp.lon + tmp.lon_delta * tmp.max_abs_change**0.5 * 0.11
    tmp['lat_plot'] = tmp.lat + tmp.dir_prob_seat_change * tmp.max_abs_change**0.5 * 0.035

    tmp['constit_page_url'] = tmp.apply(lambda row: f"/postcode?postcode_choice={row['constit'].upper().replace(' ','+')}#election", axis=1)

    max_triangle_size = 200 if mobile_mode else 800

    points = alt.Chart(tmp).mark_point(opacity=1, stroke='silver', strokeWidth=1.5, filled=True).encode(
        longitude='lon_plot:Q',
        latitude='lat_plot:Q',
        size=alt.Size('abs_prob_seat_change:Q', scale=alt.Scale(domain=[0,1], range=[25, max_triangle_size], type='pow', exponent=1.5), legend=None),
        shape=alt.Shape('dir_prob_seat_change:N', scale=alt.Scale(domain=[-1,1], range=['triangle-down','triangle-up']), legend=None),
        color = alt.Color('party_short', 
                    scale=alt.Scale(
                        domain=party_colours[party_colours.party_name.isin(elct_fcst_seat_deltas.party_short)]['party_name'].tolist(), 
                        range=party_colours[party_colours.party_name.isin(elct_fcst_seat_deltas.party_short)]['colour'].tolist()
                        ), legend=None),
        tooltip = 'tooltip_text:N'
    )
    if not mobile_mode:
        points = points.encode(href = 'constit_page_url:N')

    plot = (map + points).properties(
        width='container',
        height=450
    )    

    return plot.to_json()

#Election most seats ring plot
@app.route('/data/elct_cw_most_seats_plot_web')
def elct_cw_most_seats_fn_web():
    return elct_cw_most_seats_fn()
@app.route('/data/elct_cw_most_seats_plot_mobile')
def elct_cw_most_seats_fn_mobile():
    return elct_cw_most_seats_fn(mobile_mode = True)

def elct_cw_most_seats_fn(mobile_mode = False):
    plot = alt.Chart(biggest_party_fracs).mark_arc(opacity=0.8, stroke='black', strokeWidth=1.5,
        innerRadius = 50 if mobile_mode else 80, 
        ).encode(
        theta=alt.Theta(field="frac_biggest", type="quantitative"),
        color=alt.Color('party:N', scale=alt.Scale(range=['maroon','darkgreen','lavender'], 
            domain=['DUP','Sinn Fein','Tie']),
            legend=alt.Legend(title='', labelFontSize=13, orient='bottom')),
        tooltip='tooltip_text:N',
    ).properties(
        width = 'container',
        height = 180 if mobile_mode else 250,
        background = 'none',
        padding=15
    )

    return plot.to_json()

@app.route('/robots.txt', methods=['GET'])
def static_from_root():
    return send_from_directory(app.static_folder, request.path[1:])


if __name__ == '__main__':
    if getpass.getuser() == 'david':
        app.run(debug=True)
    else:
        app.run(debug=False)  #don't need this for PythonAnywhere?


