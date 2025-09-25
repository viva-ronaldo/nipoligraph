import json, re, requests
import numpy as np
import pandas as pd
import boto3
from nltk.tokenize import WordPunctTokenizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel

#
UNIONIST_PARTIES = ['Ulster Unionist Party', 'Traditional Unionist Voice', 'Democratic Unionist Party']
NATIONALIST_PARTIES = ['Sinn Féin', 'Social Democratic and Labour Party']
OTHER_PARTIES = ['Alliance Party', 'Green Party', 'People Before Profit Alliance', 'Independent']

MOTION_CARRIED_PHRASES = ['The Amendment Was Therefore Agreed',
                          'The Motion Was Carried',
                          'The Motion Was Carried By Cross Community Consent',
                          'The Motion, As Amended, Was Carried',
                          'The Motion Was Carried By Simple Majority']
MOTION_FAILED_PHRASES = ['The Motion Was Negatived',
                         'The Amendment Therefore Fell']
#


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

def score_contribs_with_lda(contribs_in_file, scored_contribs_out_file, lda_model, stopwords_filepath):
    print('\nScoring plenary contribs with LDA topic model')

    contribs = pd.read_feather(contribs_in_file)
    
    #First catch special uses of House and Chamber before casing
    contribs['text_proc'] = contribs.contrib
    for fake_house_phrase in ['the House', 'the Chamber', 'this House', 'this Chamber']:
        contribs['text_proc'] = contribs.text_proc.str.replace(fake_house_phrase, '_house_token_')  #to distinguish from housing
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
    contribs['text_proc'] = contribs.text_proc.str.replace('go raibh', 'go_raibh')
    contribs['text_proc'] = contribs.text_proc.str.replace('-', '_')
    contribs['text_proc'] = contribs.text_proc.apply(lambda t: re.sub('£[\\.\\,\\dm]*', '_price_token_', t))
    contribs['text_proc'] = contribs.text_proc.apply(lambda t: re.sub('20[\\d\\-]{2,7}', '_year_token_', t))
    #Remove 'leave out all after XXX and insert'
    contribs['text_proc'] = contribs.text_proc.apply(lambda t: re.sub('leave out all after .* and insert:?\n\n\"', ' ', t))
    
    #Tokenize documents
    #stopWords = set(stopwords.words('english'))
    if 's3' in stopwords_filepath:
        s3 = boto3.client('s3')
        bucket_name = stopwords_filepath.replace('s3://', '').split('/')[0]
        response = s3.get_object(Bucket=bucket_name, Key=stopwords_filepath.split('/')[-1])
        stopWords = set(response['Body'].read().decode('utf-8'))
    else:
        with open(stopwords_filepath, 'r') as f:
            stopWords = set([w.strip() for w in f.readlines()])
    
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
    corpus = [lda_model['dictionary'].doc2bow(doc) for doc in contribs.tokenized_text.tolist()]
    
    #assign topics
    contribs['topic_num'] = [assign_most_likely_topic(l, topic_nums_to_drop=lda_model['topic_nums_to_drop']) \
                             for l in lda_model['topic_model'].get_document_topics(corpus, minimum_probability=0.1)]
    contribs['topic_name'] = contribs.topic_num.apply(lambda n: lda_model['topic_name_dict'][n])
    
    #save
    contribs[['speaker', 'session_id', 'topic_name']].to_csv(scored_contribs_out_file, index=False)
    print(f"- Done: {contribs.shape[0]} contributions")


# Assign bloc vote labels - all AYE, NO, ABSTAINED, or split
def get_bloc_vote_value(bloc_df, unanimous_threshold=0.9):
    bloc_vote_counts = bloc_df.Vote.value_counts()
    if bloc_vote_counts.max() / bloc_vote_counts.sum() >= unanimous_threshold:
        bloc_vote = bloc_vote_counts.sort_values().index[-1]
    else:
        bloc_vote = 'ABSTAINED' if bloc_vote_counts.shape[0] == 0 else 'split'
    return bloc_vote

# Get bloc vote values by votes_df.EventId (i.e. one row per vote)
def analyse_votes_by_bloc(votes_filepath,
                          vote_results_filepath,
                          mla_ids_filepath,
                          party_names_translation_filepath,
                          v_comms_out_filepath):
    '''
    TODO
    '''
    mla_ids = pd.read_csv(mla_ids_filepath, dtype = {'PersonId': object})
    party_group_dict = dict(
        [(p, 'Unionist') for p in UNIONIST_PARTIES]
        + [(p, 'Nationalist') for p in NATIONALIST_PARTIES]
        + [(p, 'Other') for p in OTHER_PARTIES]
    )
    mla_ids['PartyGroup'] = mla_ids.PartyName.apply(lambda p: party_group_dict[p])
    #Subset to MLAs for now
    mla_ids = mla_ids[mla_ids.role == 'MLA']
    
    if 's3' in party_names_translation_filepath:
        s3 = boto3.client('s3')
        bucket_name = party_names_translation_filepath.replace('s3://', '').split('/')[0]
        response = s3.get_object(Bucket=bucket_name, Key=party_names_translation_filepath.split('/')[-1])
        party_names_translation = json.loads(response['Body'].read().decode('utf-8'))
    else:
        with open(party_names_translation_filepath, 'r') as f:
            party_names_translation = json.load(f)

    # Combine vote results and vote detail tables, in full
    vote_results_df = pd.read_feather(vote_results_filepath)
    vote_results_df = vote_results_df.merge(mla_ids[['PersonId', 'PartyName']], 
        on='PersonId', how='left')
    vote_results_df = vote_results_df[vote_results_df.PartyName.notnull()]  #drop a few with missing member and party names
    vote_results_df['PartyName'] = vote_results_df.PartyName.apply(lambda p: party_names_translation[p])
    
    votes_df = pd.read_feather(votes_filepath)
    votes_df = votes_df.merge(vote_results_df, on='EventId', how='inner')
    # Check no voters unknown to mla_ids have appeared
    assert set(votes_df.PersonId.unique()).issubset(mla_ids.PersonId), 'Unknown voter ID in votes_df'
    votes_df = votes_df.merge(mla_ids[['PersonId', 'normal_name']], on='PersonId', how='inner')
    #Check no new Outcome wordings have been added
    unexpected_outcome_phrases = set(votes_df.Outcome.unique()).difference(
        MOTION_CARRIED_PHRASES + MOTION_FAILED_PHRASES)
    assert unexpected_outcome_phrases == set(), \
        f"New/unexpected Outcome wording in vote_results_df: {', '.join(unexpected_outcome_phrases)}"
    
    votes_df['DivisionDate'] = pd.to_datetime(votes_df['DivisionDate'], utc=True)
    votes_df = votes_df.sort_values('DivisionDate')
    #now simplify to print nicer
    votes_df['DivisionDate'] = votes_df['DivisionDate'].dt.date
    #To pass all votes list, create a column with motion title and url 
    #  joined by | so that I can split on this inside the datatable
    votes_df['motion_plus_url'] = votes_df.apply(
        lambda row: f"{row['Title']}|https://aims.niassembly.gov.uk/plenary/details.aspx?&ses=0&doc={row['DocumentID']}&pn=0&sid=vd",
        axis=1)

    # Do the group vote calculations
    v_comms = []
    for v_id in votes_df.EventId.unique():
        tmp = votes_df.loc[votes_df.EventId==v_id]
        
        #Tabler(s) are not known for EventId 1084 
        if tmp.tabler_personIDs.iloc[0] == '':
            continue
        
        vote_date = str(tmp.DivisionDate.iloc[0])
        vote_subject = tmp.motion_plus_url.iloc[0]
        vote_result = 'PASS' if tmp.Outcome.iloc[0] in MOTION_CARRIED_PHRASES else 'FAIL'
        vote_tabler_group = tmp.tabler_personIDs.iloc[0].split(';')
        vote_tabler_group = [mla_ids.loc[mla_ids.PersonId==x, 'PartyGroup'].iloc[0] for x in vote_tabler_group]
        vote_tabler_group = vote_tabler_group[0] if len(set(vote_tabler_group)) == 1 else 'Mixed'
            
        uni_bloc_vote = get_bloc_vote_value(tmp[tmp.Designation == 'Unionist'])
        nat_bloc_vote = get_bloc_vote_value(tmp[tmp.Designation == 'Nationalist'])
        alli_vote = get_bloc_vote_value(tmp[tmp.PartyName == 'Alliance'])
        green_vote = get_bloc_vote_value(tmp[tmp.PartyName == 'Green'])
        
        # Thought about doing DUP+SF group here too but not implemented yet
        
        v_comms.append((v_id, vote_date, vote_subject, vote_tabler_group, vote_result,
                        uni_bloc_vote, nat_bloc_vote, alli_vote, green_vote))
    
    v_comms = pd.DataFrame(v_comms, columns=['EventId', 'vote_date', 'vote_subject',
                                             'vote_tabler_group', 'vote_result',
                                             'uni_bloc_vote', 'nat_bloc_vote',
                                             'alli_vote', 'green_vote'])

    # There was a unionist/nationalist split if each bloc voted as one but they went differently to one another
    v_comms['uni_nat_split'] = (v_comms.uni_bloc_vote != 'split') & \
        (v_comms.nat_bloc_vote != 'split') & (v_comms.nat_bloc_vote != v_comms.uni_bloc_vote)
    v_comms['uni_nat_split'] = v_comms.uni_nat_split.apply(lambda b: 'Yes' if b else 'No')

    v_comms = v_comms.sort_values('vote_date', ascending=False).reset_index(drop=True)

    # Make the whole file (it's not many rows) new each time.
    v_comms.to_csv(v_comms_out_filepath, index=False)

# Bluesky

def fetch_bluesky_posts(account_id, cursor=None):
    """Fetch posts for a given account ID from the BlueSky API."""
    API_BASE_URL = "https://public.api.bsky.app/xrpc"
    url = f"{API_BASE_URL}/app.bsky.feed.getAuthorFeed"
    headers = {} #"Authorization": f"Bearer {API_TOKEN}"}
    params = {"actor": account_id, 'limit': 50}
    # format must be '2024-11-20T19:04:04.583Z'; returns anything before this, exclusive
    if cursor is not None:
        params['cursor'] = cursor

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch posts for {account_id}: {response.status_code}")
        return []

    data = response.json()
    posts = []
    for item in data.get("feed", []):
        is_repost = 'reason' in item.keys()

        post = item['post']

        if not is_repost and post['likeCount'] > 0:
            url_likes = f"{API_BASE_URL}/app.bsky.feed.getLikes"
            params_likes = {'uri': post['uri']}
            response_likes = requests.get(url_likes, headers={}, params=params_likes)
            post_likes = [l['actor']['handle'] for l in response_likes.json()['likes']]
        else:
            post_likes = []

        if not is_repost and post['repostCount'] > 0:
            url_reposts = f"{API_BASE_URL}/app.bsky.feed.getRepostedBy"
            params_reposts = {'uri': post['uri']}
            response_reposts = requests.get(url_reposts, headers={}, params=params_reposts)
            post_reposts = [l['handle'] for l in response_reposts.json()['repostedBy']]
        else:
            post_reposts = []
        

        posts.append({
            "account_id": account_id,
            "date": post['record']['createdAt'],
            "uri": post['uri'],
            "is_repost": is_repost,
            "author": post['author']['handle'],
            "text": post['record']['text'],
            "reply_count": post['replyCount'],
            "repost_count": post['repostCount'],
            "reposted_by": post_reposts,
            "like_count": post['likeCount'],
            "liked_by": post_likes,
            "quote_count": post['quoteCount'],
        })
    return posts

def append_bluesky_posts_to_feather(new_data, file_path):
    """Append new data to an existing Feather file or create a new one."""
    try:
        # Load existing data
        df_existing = pd.read_feather(file_path)
        print(f'{len(df_existing)} Bluesky rows existing')
    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame
        df_existing = pd.DataFrame()

    # Convert new data to DataFrame
    df_new = pd.DataFrame(new_data)

    # Combine and save to Feather
    df_combined = (
        pd.concat([df_existing, df_new], ignore_index=True)
        .drop_duplicates(subset=['uri'])
        .reset_index(drop=True)
    )
    print(f'Increased to {len(df_combined)} Bluesky rows')
    df_combined.to_feather(file_path)
