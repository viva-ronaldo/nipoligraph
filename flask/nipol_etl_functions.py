import datetime, feather, json, os, pickle, requests
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA

def get_plenary_contribs_colour_dict():
    """
    Use the colours we get from 'tableau20' but make repeatable and for reuse on indiv page
    """
    return {
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

def get_diary_colour_dict():
    """ """
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
    return diary_colour_dict

def get_parties_colour_dict(data_dir, config):
    """ """
    party_colours = pd.read_csv(data_dir + config['DATA_PARTY_COLOURS'])
    col_corrects = {'yellow3': '#e6c300', 'green2': 'chartreuse'}
    party_colours['colour'] = party_colours['colour'].apply(
        lambda c: c if c not in col_corrects.keys() else col_corrects[c]
    )
    return party_colours

def get_member_name_fix_dict(data_dir, config):
    """
    Fix awkward names in apg and comm attendees which don't use PersonId so can fall out of sync when AIMS changes a name
    This needs to list any new or old names used for an MLA and point to their current PersonId name; 
      and need to include inactive in mla_ids
    """
    mla_ids = pd.read_csv(data_dir + config['DATA_MEMBER_IDS'], dtype = {'PersonId': object})
    mla_name_fix_dict = {
        'Lord Elliott of Ballinamallard': mla_ids[mla_ids.PersonId=='128'].iloc[-1].normal_name,
        'Lord Tom Elliott of Ballinamallard': mla_ids[mla_ids.PersonId=='128'].iloc[-1].normal_name,
        'Tom Elliott': mla_ids[mla_ids.PersonId=='128'].iloc[-1].normal_name,
    }
    return mla_name_fix_dict

def assign_assembly_session_name(date_string):
    """ Assign an AIMS date (e.g., TabledDate) to a session name"""
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
    elif date_string < '2027-06-01':
        session_name = '2022-2027'
    else:
        session_name = '2027-2032'
    return session_name
#There were some questions submitted in the 2017-2019 period but don't plot this session

def get_current_avg_poll_pct(polls_df, elections_df, party, current_dt, time_power = 4, assembly_equiv_sample_size = 50000, general_equiv_sample_size = 500):
    """ TODO """
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

def convert_fraction_to_words(fraction):
    """ Convert a numerical fractional chance to a description """
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
    """ Make a description string for `row` containing forecast of party fractional seats won """
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

def find_profile_photo(person_is_mla, person_id, normal_name, member_other_photo_links, mp_api_number=None):
    """ """
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

def load_blog_item_table(data_dir, post_name, table_number):
    """ """
    table_df = pd.read_csv(data_dir + f'blog-data_{post_name}_table-{table_number}.csv')
    table_df_list = [e[1].values.tolist() for e in table_df.iterrows()]
    return table_df_list

def get_rank_category_index(value, rank_split_points):
    """ Find bucket number for value; there is probably a numpy function for this """
    for i in range(len(rank_split_points)):
        if value <= rank_split_points[i]:
            return i
    else:
        return len(rank_split_points)


# --- Some data load functions ---

# Ids
def load_member_ids(data_dir, config, active_only=True):
    """
    """
    mla_ids = pd.read_csv(data_dir + config['DATA_MEMBER_IDS'], dtype = {'PersonId': object})
    
    #If have too many non-MLA/MPs, could become unreliable over time, so limit to active here
    if active_only:
        mla_ids = mla_ids[mla_ids.active==1]
    
    mla_ids['fullname'] = mla_ids['MemberFirstName'] + ' ' + mla_ids['MemberLastName']  # slightly different from normal_name

    #Add email address
    mla_ids = (
        mla_ids
        .merge(pd.read_csv(data_dir + config['DATA_MLA_EMAILS'], dtype = {'PersonId': object}),
            on='PersonId', how='left')
        .merge(pd.read_csv(data_dir + config['DATA_MP_EMAILS']), on='normal_name', how='left')
        .fillna({'AssemblyEmail': 'none', 'WestminsterEmail': 'none'})
    )

    with open(data_dir + config['DATA_PARTY_GROUPS'], 'r') as f:
        party_group_dict = json.load(f)
    #Do this before turning PartyName to short form
    mla_ids['PartyGroup'] = mla_ids.PartyName.apply(lambda p: party_group_dict[p])

    #Handle the two forms of some party names
    #with open(data_dir + 'party_names_translation.json', 'r') as f:
    with open(data_dir + config['DATA_PARTY_NAMES_TRANSLATION_SHORT'], 'r') as f:
        party_names_translation = json.load(f)
    with open(data_dir + config['DATA_PARTY_NAMES_TRANSLATION_LONG'], 'r') as f:
        party_names_translation_long = json.load(f)

    mla_ids['PartyName_long'] = mla_ids.PartyName.apply(lambda p: party_names_translation_long[p])
    mla_ids['PartyName'] = mla_ids.PartyName.apply(lambda p: party_names_translation[p])

    return mla_ids, party_names_translation, party_names_translation_long

# Assembly ids
def load_minister_roles(data_dir, config, mla_ids):
    """ """
    mla_minister_roles = (
        pd.read_csv(data_dir + config['DATA_ASSEMBLY_MINISTER_ROLES'], dtype={'PersonId': object})
        .merge(mla_ids[['PersonId', 'normal_name']], on='PersonId', how='inner')
    )
    mla_minister_roles = {i[1]['normal_name']: i[1]['AffiliationTitle'] for i in mla_minister_roles.iterrows()}
    return mla_minister_roles

def load_and_process_assembly_committees(data_dir, config, mla_ids):
    """
    Read Assembly committee roles, meeting attendance, and agendas files
    and combine to produce meeting table with agenda and attendees.
    """
    mla_name_fix_dict = get_member_name_fix_dict(data_dir, config)

    committee_roles = (
        pd.read_csv(data_dir + config['DATA_ASSEMBLY_COMMITTEE_MEMBERSHIP'], dtype={'PersonId': object})
        .merge(mla_ids[['PersonId', 'normal_name']], on='PersonId', how='inner')
    )
    committee_meeting_attendance = (
        pd.read_csv(data_dir + config['DATA_ASSEMBLY_COMMITTEE_MEETING_ATTENDANCE'])
        .assign(member = lambda df: df.member.str.replace('Mr |Mrs |Ms |Miss |Dr |Sir | OBE| MBE| CBE| MC', '', regex=True))
        .assign(member = lambda df: df.member.apply(lambda m: mla_name_fix_dict.get(m, m)))
        .rename(columns={'member': 'fullname'})
        .merge(mla_ids[['fullname', 'normal_name', 'MemberLastName']], on='fullname', how='inner')
        .drop(columns='fullname')
        .assign(meeting_date = lambda df: pd.to_datetime(df.meeting_date, format='%d/%m/%Y'))
    )

    committee_attendees_agg = (
        committee_meeting_attendance
        .query('attended')
        .groupby('meeting_id')
        .agg(attendees = ('MemberLastName', lambda m: ', '.join(m)))
        .assign(attendees = lambda df: df.attendees.fillna('-'))
    )

    committee_meeting_agendas = pd.read_csv(data_dir + config['DATA_ASSEMBLY_COMMITTEE_MEETING_AGENDAS'])

    def truncate_agenda_list(agenda_items):
        agenda_items = agenda_items.tolist()
        if len(agenda_items) > 5:
            agenda_items = agenda_items[:4] + [f'<i>and {len(agenda_items)-4} more items</i>']
        return '<ul><li>' + '</li><li>'.join(agenda_items) + '</li></ul>'
    
    committee_agendas_agg = (
        committee_meeting_agendas
        .assign(meeting_date = lambda df: pd.to_datetime(df.meeting_date, format='%d/%m/%Y'))
        .groupby(['meeting_id', 'committee_name', 'meeting_date'], as_index=False)
        .agg(agenda_list = ('agenda_item', truncate_agenda_list))
        .merge(committee_attendees_agg, on='meeting_id', how='left')
        .fillna({'attendees': 'Unknown'})
        .sort_values(['meeting_date', 'meeting_id'], ascending=False)
        .assign(meeting_date_and_url = lambda df: df.apply(lambda row: f"{row.meeting_date.strftime('%d %B %Y')}|https://aims.niassembly.gov.uk/committees/meetings.aspx?cid=0&mid={row.meeting_id}", axis=1))
        .filter(['meeting_date_and_url', 'committee_name', 'agenda_list', 'attendees'])
    )

    return committee_roles, committee_meeting_attendance, committee_agendas_agg

def load_mla_interests(data_dir, config, mla_ids):
    return (
        pd.read_csv(data_dir + config['DATA_ASSEMBLY_REGISTERED_INTERESTS'], dtype={'PersonId': object})
        .merge(mla_ids[['PersonId', 'normal_name']], on='PersonId', how='inner')
    )

def load_allpartygroup_memberships(data_dir, config, mla_ids):
    mla_name_fix_dict = get_member_name_fix_dict(data_dir, config)
    return (
        pd.read_csv(data_dir + config['DATA_ASSEMBLY_ALLPARTYGROUP_MEMBERSHIP'])
        .assign(member = lambda df: df.member.apply(lambda m: mla_name_fix_dict.get(m, m)))
        .rename(columns={'member': 'fullname'})
        .merge(mla_ids[['fullname', 'normal_name']], on='fullname', how='inner')
        .drop(columns='fullname')
        .assign(apg_name = lambda df: df.apg_name.str.replace('All-Party [gG]roup on ', '', regex=True))
    )

def load_assembly_diary(data_dir, config):
    """ """
    diary_colour_dict = get_diary_colour_dict()

    if os.stat(data_dir + config['DATA_ASSEMBLY_DIARY_EVENTS']).st_size > 10:
        diary_df = (
            pd.read_csv(data_dir + config['DATA_ASSEMBLY_DIARY_EVENTS'], sep='|')
            #Use StartTime to get the date to avoid problems with midnight +/-1h BST
            .assign(EventPrettyDate = lambda df: pd.to_datetime(df['StartTime'], utc=True).dt.strftime('%A, %-d %B'))
            .assign(EventName = lambda df: df.apply(
                lambda row: row['OrganisationName']+' Meeting' if row['EventType']=='Committee Meeting' else row['EventType'], 
                axis=1)
            )
            .assign(EventHTMLColour = lambda df: df.EventName.apply(lambda e: diary_colour_dict[e]))
        )
        #Excluding events that have now happened will happen in in assembly.html function    
    else:
        diary_df = pd.DataFrame({'EventName': [], 'EndTime': []}, dtype=str)
    
    return diary_df

# Social media
def load_and_process_twitter(data_dir, config, mla_ids):
    """
    Hard code data and ids files as don't expect to use or update this
    """
    with open(data_dir + config['DATA_PARTY_NAMES_TRANSLATION_SHORT'], 'r') as f:
        party_names_translation = json.load(f)

    #tweets_df = feather.read_dataframe(data_dir + 'mlas_2019_tweets_apr2019min_to_present_slim.feather')
    tweets_df = pd.concat([
        feather.read_dataframe(data_dir + 'tweets_slim_apr2019min_to_3jun2021.feather'),
        feather.read_dataframe(data_dir + 'tweets_slim_4jun2021_to_present.feather')
    ])
    tweets_df = tweets_df.rename(columns={'is_retweet': 'is_repost'})
    twitter_ids = pd.read_csv(data_dir + 'politicians_twitter_accounts_ongoing.csv',
        dtype = {'user_id': object})
    #tweets_df = tweets_df.merge(twitter_ids[['user_id','mla_party','mla_name']].rename(index=str, columns={'mla_name':'normal_name'}), 
    tweets_df = tweets_df.merge(twitter_ids[['user_id', 'mla_party', 'normal_name']],
        on='user_id', how='left')
    tweets_df['mla_party'] = tweets_df.mla_party.apply(lambda p: party_names_translation[p])
    #Filter to 1 July onwards - fair comparison for all
    tweets_df = tweets_df[tweets_df.created_ym >= '202007']
    tweets_df = tweets_df[tweets_df['normal_name'].isin(mla_ids['normal_name'])]
    
    tweets_df['post_type'] = tweets_df.is_retweet.apply(lambda b: 'repost' if b else 'original')
    tweets_df['involves_quote'] = tweets_df.quoted_status_id.apply(lambda s: s is not None)

    tweets_df['created_at_week'] = tweets_df['created_at'].dt.isocalendar().week
    #tweets_df['created_at_week'] = tweets_df['created_at'].dt.week
    #early Jan can be counted as week 52 or 53 by pd.week - messes things up
    tweets_df.loc[(tweets_df.created_at_week >= 52) & (tweets_df.created_at.dt.day <= 7), 'created_at_week'] = 1
    tweets_df['created_at_yweek'] = tweets_df.apply(
        lambda row: '{:s}-{:02g}'.format(row['created_ym'][:4], row['created_at_week']), axis=1)

    last_5_yweeks_tweets = tweets_df.created_at_yweek.sort_values().unique()[-5:]
    print('Retweets are counted over', last_5_yweeks_tweets)

    tweet_sentiment = pd.read_csv(data_dir + 'vader_scored_tweets_apr2019min_to_present.csv', dtype={'status_id': object})
    tweets_df = tweets_df.merge(tweet_sentiment, on='status_id', how='left')

    #Which people tweet the most
    top_tweeters = (
        tweets_df[tweets_df.created_at_yweek.isin(last_5_yweeks_tweets)]
        .groupby(['normal_name', 'mla_party', 'post_type'], as_index=False)
        .agg(n_posts = ('status_id', len))
        .assign(tooltip_text = lambda df: df.apply(
            lambda row: f"{row['normal_name']} ({row['mla_party']}): {row['n_posts']} {'original posts' if row['post_type'] == 'original' else 'reposts'}", axis=1)
        )
    )
    #(now filter to top 10/15 in function below)

    member_retweets = (
        tweets_df[(~tweets_df.is_retweet) & (tweets_df.created_at_yweek.isin(last_5_yweeks_tweets))]
        .groupby(['normal_name', 'mla_party'], as_index=False)
        .agg(n_original_posts = ('status_id', len), reposts_per_post = ('repost_count', 'mean'))
        .query('n_original_posts >= 10')
        .sort_values('reposts_per_post', ascending=False)
        .assign(tooltip_text = lambda df: df.apply(
            lambda row: f"{row['normal_name']}: {row['reposts_per_post']:.1f} repost per original post (from {row['n_original_posts']} posts)", axis=1)
        )
    )

    member_tweet_sentiment = tweets_df[(tweets_df.created_at_yweek.isin(last_5_yweeks_tweets)) & (tweets_df.sentiment_vader_compound.notnull())]\
        .groupby(['normal_name', 'mla_party'])\
        .agg({'sentiment_vader_compound': np.mean, 'status_id': len}).reset_index()\
        .query('status_id >= 10').sort_values('sentiment_vader_compound', ascending=False)
    #Normalise here - only using for one plot (and rankings on indiv pages)
    member_tweet_sentiment['sentiment_vader_compound'] = member_tweet_sentiment['sentiment_vader_compound'] - member_tweet_sentiment['sentiment_vader_compound'].mean()
    if len(member_tweet_sentiment) > 0:
        member_tweet_sentiment['tooltip_text'] = member_tweet_sentiment.apply(
            lambda row: f"{row['normal_name']}: mean score rel. to avg. = {row['sentiment_vader_compound']:+.2f} ({row['status_id']} posts)", 
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

    return tweets_df, top_tweeters, member_retweets, member_tweet_sentiment

def load_and_process_bluesky(data_dir, config, mla_ids):
    """
    Load the posts file and return raw data, top posters, reposts, and sentiment dfs
    """
    with open(data_dir + config['DATA_PARTY_NAMES_TRANSLATION_SHORT'], 'r') as f:
        party_names_translation = json.load(f)

    bs_posts = (
        pd.read_feather(data_dir + config['DATA_BLUESKY_POSTS'])
        .rename(columns={'uri': 'status_id'})
        .merge(mla_ids[['normal_name', 'PartyName']], on='normal_name', how='inner')
        .assign(mla_party = lambda df: df.PartyName.apply(lambda p: party_names_translation[p]),
            post_type = lambda df: df.is_repost.apply(lambda b: 'repost' if b else 'original'),
            involves_quote = False,
            created_at_dt = lambda df: pd.to_datetime(df['date']),
            created_at_week = lambda df: df.created_at_dt.dt.isocalendar().week,
            created_at_yweek = lambda df: df.apply(
                lambda row: '{:04g}-{:02g}'.format(row.created_at_dt.year, row.created_at_week), axis=1),
            screen_name = lambda df: df.account_id
            )
    )
    #early Jan can be counted as week 52 or 53 by pd.week - messes things up
    bs_posts.loc[(bs_posts.created_at_week >= 52) & (bs_posts.created_at_dt.dt.day <= 7), 'created_at_week'] = 1
    most_recent_post_dt = bs_posts.created_at_dt.max()

    #Which people post the most - last year
    top_posters = (
        bs_posts[bs_posts.created_at_dt >= most_recent_post_dt - datetime.timedelta(days=365)]
        .groupby(['normal_name', 'mla_party', 'post_type'], as_index=False)
        .agg(n_posts = ('status_id', len))
        .assign(tooltip_text = lambda df: df.apply(
            lambda row: f"{row['normal_name']} ({row['mla_party']}): {row['n_posts']} {'original posts' if row['post_type'] == 'original' else 'reposts'}", axis=1)
        )
    )

    #Who gets the most reposts (from anyone) - last year, minimum 5 original posts
    member_reposts = (
        bs_posts[(~bs_posts.is_repost) & (bs_posts.created_at_dt >= most_recent_post_dt - datetime.timedelta(days=365))]
        .groupby(['normal_name', 'mla_party'], as_index=False)
        .agg(n_original_posts = ('status_id', len), reposts_per_post = ('repost_count', 'mean'))
        .query('n_original_posts >= 5')
        .sort_values('reposts_per_post', ascending=False)
        .assign(tooltip_text = lambda df: df.apply(
            lambda row: f"{row['normal_name']}: average {row['reposts_per_post']:.1f} reposts (from {row['n_original_posts']} posts)", axis=1)
        )
    )

    #Sentiment average - last year, minimum 10 posts
    member_sm_sentiment = (
        bs_posts[bs_posts.created_at_dt >= most_recent_post_dt - datetime.timedelta(days=365)]
        .groupby(['normal_name', 'mla_party'], as_index=False)
        .agg({'sentiment_vader_compound': np.mean, 'status_id': len})
        .query('status_id >= 10')
        .sort_values('sentiment_vader_compound', ascending=False)
    )
    #Normalise here - only using for one plot (and rankings on indiv pages)
    member_sm_sentiment['sentiment_vader_compound'] = member_sm_sentiment['sentiment_vader_compound'] - member_sm_sentiment['sentiment_vader_compound'].mean()
    if len(member_sm_sentiment) > 0:
        member_sm_sentiment['tooltip_text'] = member_sm_sentiment.apply(
            lambda row: f"{row['normal_name']}: mean score rel. to avg. = {row['sentiment_vader_compound']:+.2f} ({row['status_id']} posts)", 
            axis=1
        )
    else:
        member_sm_sentiment['tooltip_text'] = []

    return bs_posts, top_posters, member_reposts, member_sm_sentiment

# Assembly
def load_and_process_assembly_questions(data_dir, config, mla_ids):
    """
    """
    questions_df = (
        feather.read_dataframe(data_dir + config['DATA_ASSEMBLY_QUESTIONS'])
        .merge(mla_ids[['PersonId', 'normal_name', 'PartyName']],
            left_on='TablerPersonId', right_on='PersonId', how='left')
        # for plot facet labels
        .assign(RequestedAnswerType = lambda df: df.RequestedAnswerType.apply(
            lambda rat: 'Oral' if rat=='oral' else ('Written' if rat=='written' else rat)))
    )

    #NB important to have nunique here to avoid counting a question to FM/DFM twice
    questioners = (
        questions_df
        .groupby(['normal_name', 'PartyName', 'RequestedAnswerType'], as_index=False)
        .DocumentId.nunique()
        .rename(index=str, columns={'DocumentId': 'Questions asked', 'RequestedAnswerType': 'Question type'})  #for plot titles
    )
    #(get top 15 by each question type - now done in function below)
    if len(questioners) > 0:
        questioners['tooltip_text'] = questioners.apply(
            lambda row: f"{row['normal_name']}: {row['Questions asked']} {row['Question type'].lower()} question{('s' if row['Questions asked'] != 1 else ''):s} asked", axis=1
        )
    else:
        questioners['tooltip_text'] = []

    return questions_df, questioners

def load_and_process_assembly_answers(data_dir, config, mla_ids, mla_minister_roles):
    """
    """
    answers_df = (
        feather.read_dataframe(data_dir + config['DATA_ASSEMBLY_ANSWERS'])
        .merge(
            mla_ids[['PersonId', 'normal_name', 'PartyName']]
            .rename(index=str, columns={
                'normal_name': 'Tabler_normal_name', 
                'PersonId': 'TablerPersonId',
                'PartyName': 'Tabler_party_name'}),
            on='TablerPersonId', how='inner')
        .merge(
            mla_ids[['PersonId','normal_name','PartyName']]
            .rename(index=str, columns={
                'normal_name': 'Minister_normal_name', 
                'PersonId': 'MinisterPersonId',
                'PartyName': 'Minister_party_name'}),
            on='MinisterPersonId', how='left')
        .assign(Days_to_answer = lambda df: df.apply(lambda row: (pd.to_datetime(row['AnsweredOnDate'][:10]) - pd.to_datetime(row['TabledDate'][:10])).days, axis=1))
    )

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
    minister_time_to_answer = (
        answers_df[answers_df.MinisterTitle != 'Assembly Commission']
        .groupby(['Minister_normal_name', 'Minister_party_name'], as_index=False)
        .agg(
            median_days_to_answer = ('Days_to_answer', np.median),
            num_questions_answered = ('Days_to_answer', len)
        )
        .rename(index=str, columns={
            'median_days_to_answer': 'Median days to answer',  
            'num_questions_answered': 'Questions answered'    #for plot axis title
        })
    )
    if len(minister_time_to_answer) > 0:
        minister_time_to_answer['tooltip_text'] = minister_time_to_answer.apply(
            lambda row: f"{row['Minister_normal_name']}: median {row['Median days to answer']:g} day{('s' if row['Median days to answer'] != 1 else ''):s}", axis=1
        )
    else:
        minister_time_to_answer['tooltip_text'] = []
    minister_time_to_answer = minister_time_to_answer[minister_time_to_answer.Minister_normal_name.isin(mla_minister_roles.keys())]

    return answers_df, minister_time_to_answer

def load_and_process_assembly_votes(data_dir, config, mla_ids):
    """
    Load Assembly division votes and vote commenary file, and do PCA on votes.
    """
    with open(data_dir + config['DATA_PARTY_NAMES_TRANSLATION_SHORT'], 'r') as f:
        party_names_translation = json.load(f)

    vote_results_df = (
        feather.read_dataframe(data_dir + config['DATA_ASSEMBLY_DIVISION_VOTE_RESULTS'])
        .merge(mla_ids[['PersonId', 'PartyName']], on='PersonId', how='left')
    )
    vote_results_df = vote_results_df[vote_results_df.PartyName.notnull()] #drop a few with missing member and party names
    vote_results_df['PartyName'] = vote_results_df.PartyName.apply(lambda p: party_names_translation[p])

    votes_df = (
        feather.read_dataframe(data_dir + config['DATA_ASSEMBLY_DIVISION_VOTES'])
        .merge(vote_results_df, on='EventId', how='inner')
        .merge(mla_ids[['PersonId', 'normal_name']], on='PersonId', how='inner')
        .assign(DivisionDate = lambda df: pd.to_datetime(df['DivisionDate'], utc=True))
        .sort_values('DivisionDate')
        #now simplify to print nicer
        .assign(DivisionDate = lambda df: df['DivisionDate'].dt.date)
    )
    #To pass all votes list, create a column with motion title and url 
    #  joined by | so that I can split on this inside the datatable
    if len(votes_df) > 0:
        votes_df['motion_plus_url'] = votes_df.apply(
            lambda row: f"{row['Title']}|https://aims.niassembly.gov.uk/plenary/details.aspx?&ses=0&doc={row['DocumentID']}&pn=0&sid=vd", axis=1)
    else:
        votes_df['motion_plus_url'] = []

    #Votes PCA
    votes_df['vote_num'] = votes_df.Vote.apply(lambda v: {'NO': -1, 'AYE': 1, 'ABSTAINED': 0}[v]) 
    votes_pca_df = (
        votes_df[['normal_name','EventId','vote_num']]
        .pivot(index='normal_name',columns='EventId',values='vote_num').fillna(0)
    )
    tmp = votes_df.normal_name.value_counts()
    those_in_threshold_pct_votes = tmp[tmp >= votes_df.EventId.nunique()*config['PCA_VOTES_THRESHOLD_FRACTION']].index
    votes_pca_df = votes_pca_df[votes_pca_df.index.isin(those_in_threshold_pct_votes)]

    mla_votes_pca = PCA(n_components=2, whiten=True)  #doesn't change results but axis units closer to 1
    mla_votes_pca.fit(votes_pca_df)
    #print(my_pca.explained_variance_ratio_)
    mlas_2d_rep = pd.DataFrame({'x': [el[0] for el in mla_votes_pca.transform(votes_pca_df)],
                                'y': [el[1] for el in mla_votes_pca.transform(votes_pca_df)],
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
    v_comms = pd.read_csv(data_dir + config['DATA_ASSEMBLY_VOTES_COMMENTARY'])

    return votes_df, vote_results_df, mla_votes_pca, mlas_2d_rep, v_comms

def load_and_process_assembly_plenary_contribs(data_dir, config):
    """ """
    #Contributions - plenary sessions - don't need the raw text
    #plenary_contribs_df = feather.read_dataframe(data_dir + 'plenary_hansard_contribs_201920sessions_topresent.feather')
    #plenary_contribs_df = plenary_contribs_df[plenary_contribs_df.speaker.isin(mla_ids.normal_name)]

    scored_plenary_contribs_df = pd.read_csv(data_dir + config['DATA_ASSEMBLY_PLENARY_CONTRIBS_SCORED'])
    #scored_plenary_contribs_df = scored_plenary_contribs_df[scored_plenary_contribs_df.speaker.isin(mla_ids.normal_name)]

    plenary_contribs_topic_counts =  (
        scored_plenary_contribs_df
        .groupby('topic_name', as_index=False)
        .count()[['topic_name','session_id']]
        .rename(index=str, columns={'session_id': 'n_contribs'})
    )
    #Load model to get the topic top words
    with open(data_dir + config['MODEL_CONTRIBS_LDA'], 'rb') as f:
        lda_stuff = pickle.load(f)
    lda_topics = lda_stuff['topic_model'].show_topics(num_topics=lda_stuff['topic_model'].num_topics, num_words=5, formatted=False)
    lda_top5s = [(el[0], ', '.join([f"'{t[0]}'" for t in el[1]])) for el in lda_topics]
    lda_top5s = [(lda_stuff['topic_name_dict'][el[0]], el[1]) for el in lda_top5s]
    plenary_contribs_topic_counts = plenary_contribs_topic_counts.merge(
        pd.DataFrame(lda_top5s, columns=['topic_name', 'top5_words'])
    )
    if len(plenary_contribs_topic_counts) > 0:
        plenary_contribs_topic_counts['tooltip_text'] = plenary_contribs_topic_counts.apply(
            lambda row: f"{row['topic_name']}: strongest words are {row['top5_words']}", axis=1
        )
    else:
        plenary_contribs_topic_counts['tooltip_text'] = []

    return scored_plenary_contribs_df, lda_top5s, plenary_contribs_topic_counts

def load_and_process_assembly_contrib_emotions(data_dir, config, mla_ids):
    """ """
    #plenary contrib emotions
    emotions_df = (
        feather.read_dataframe(data_dir + config['DATA_ASSEMBLY_PLENARY_CONTRIBS_AVERAGE_EMOTIONS'])
        #TODO empty file? It should get remade next update when there are some entries
        .merge(mla_ids[['normal_name', 'PartyName']], left_on='speaker', right_on='normal_name', how='inner')
    )
    emotions_party_agg = (
        emotions_df
        .groupby(['PartyName', 'emotion_type'], as_index=False)
        .apply(lambda x_grouped: np.average(x_grouped['ave_emotion'], weights=x_grouped['word_count']))
        .rename(index=str, columns={None: 'ave_emotion'})
    )
    return emotions_df, emotions_party_agg

# Historical assembly
def load_and_process_all_historical_assembly(data_dir, config, lda_top5s):
    """ Load multiple historical Assembly tables """

    with open(data_dir + config['DATA_PARTY_GROUPS'], 'r') as f:
        party_group_dict = json.load(f)
    with open(data_dir + config['DATA_PARTY_NAMES_TRANSLATION_SHORT'], 'r') as f:
        party_names_translation = json.load(f)
    with open(data_dir + config['DATA_PARTY_NAMES_TRANSLATION_LONG'], 'r') as f:
        party_names_translation_long = json.load(f)
    valid_session_names = config['ALL_ASSEMBLY_SESSIONS']

    hist_mla_ids = pd.concat([
        feather.read_dataframe(data_dir + 'hist_mla_ids_by_session.feather'),
        feather.read_dataframe(data_dir + 'hist_20202022_mla_ids_by_session.feather')
        ])
    assert len(hist_mla_ids) == 1616

    hist_mla_ids['PartyGroup'] = hist_mla_ids.PartyName.apply(lambda p: party_group_dict[p])
    hist_mla_ids['PartyName_long'] = hist_mla_ids.PartyName.apply(lambda p: party_names_translation_long[p])
    hist_mla_ids['PartyName'] = hist_mla_ids.PartyName.apply(lambda p: party_names_translation[p])

    hist_questions_df = pd.concat([
        feather.read_dataframe(data_dir + 'historical_niassembly_questions_asked.feather'),
        feather.read_dataframe(data_dir + 'historical_20202022_niassembly_questions_asked.feather')
        ])
    assert len(hist_questions_df) >= 170_000
    hist_questions_df['session_name'] = hist_questions_df.TabledDate.apply(assign_assembly_session_name)
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
    assert len(hist_answers_df) > 150_000
    hist_answers_df['session_name'] = hist_answers_df.AnsweredOnDate.apply(assign_assembly_session_name)
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
    assert len(hist_votes_df) == 1052
    hist_votes_df = hist_votes_df.rename(index=str, columns={'EventID':'EventId'})

    hist_vote_results_df = pd.concat([
        feather.read_dataframe(data_dir + 'historical_division_vote_results.feather'),
        feather.read_dataframe(data_dir + 'historical_20202022_division_vote_results.feather')
        ])
    assert len(hist_vote_results_df) >= 80_000
    #Reorder operations from above
    hist_votes_df = hist_votes_df.merge(hist_vote_results_df, on='EventId', how='inner')

    hist_votes_df['session_name'] = hist_votes_df.DivisionDate.apply(assign_assembly_session_name)
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
        those_in_threshold_pct_votes = tmp[tmp >= hist_votes_one_session.EventId.nunique()*config['PCA_VOTES_THRESHOLD_FRACTION']].index
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
    assert len(hist_v_comms) == 1062
    assert sum(hist_v_comms.session_name.notnull()) == 1062
    print('Done historical votes')

    hist_plenary_contribs_df = pd.concat([
        pd.read_csv(data_dir + 'hist_lda_scored_plenary_contribs.csv'),
        pd.read_csv(data_dir + 'hist_20202022_lda_scored_plenary_contribs.csv')
        ])
    assert len(hist_plenary_contribs_df) >= 90000
    hist_plenary_contribs_df['session_name'] = hist_plenary_contribs_df.PlenaryDate.apply(assign_assembly_session_name)

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
    assert len(hist_emotions_df) >= 5000
    hist_emotions_df = hist_emotions_df.merge(hist_mla_ids[['normal_name','PartyName','session_name']], 
        left_on=['speaker','session_name'], 
        right_on=['normal_name','session_name'], 
        how='inner')
    hist_emotions_party_agg = hist_emotions_df.groupby(['session_name','PartyName','emotion_type']).apply(
        lambda x_grouped: np.average(x_grouped['ave_emotion'], weights=x_grouped['word_count']))\
        .reset_index().rename(index=str, columns={0:'ave_emotion'})

    return (
        hist_mla_ids,
        hist_questions_df, hist_questioners, hist_answers_df, hist_minister_time_to_answer,
        hist_votes_df, hist_votes_pca_res, hist_v_comms,
        hist_plenary_contribs_df, hist_plenary_contribs_topic_counts, hist_emotions_party_agg
    )


# News

# TODO check the yweek bit again
def load_and_process_news_data(data_dir, config, mla_ids, news_volume_average_window_weeks=7):
    """ """
    # Only really need any that will appear in the top sources plot.
    with open(data_dir + config['DATA_NEWS_SOURCE_PPRINT_DICT'], 'r') as f:
        news_source_pprint_dict = json.load(f)

    news_df = (
        pd.concat([
            #feather.read_dataframe(data_dir + 'newscatcher_articles_slim_w_sentiment_julaugsep2020.feather'),
            #feather.read_dataframe(data_dir + config['DATA_NEWS_NEWSCATCHER']),
            feather.read_dataframe(data_dir + config['DATA_NEWS_WORLDNEWS'])
        ]).drop_duplicates()
        .query("published_date >= '2023-12-01'")
        .assign(source = lambda df: df.source.apply(lambda s: news_source_pprint_dict.get(s, s)))
        .assign(published_date = lambda df: pd.to_datetime(df.published_date))
        .assign(published_date_week = lambda df: df.published_date.dt.isocalendar().week)
        .assign(published_date_year = lambda df: df.published_date.dt.year)
    )

    #early Jan can be counted as week 53 by pd.week - messes things up
    news_df.loc[(news_df.published_date_week==53) & (news_df.published_date.dt.day <= 7), 'published_date_week'] = 1

    if len(news_df) > 0:
        news_df['published_date_yweek'] = news_df.apply(
            lambda row: '{:04g}-{:02g}'.format(row['published_date_year'], row['published_date_week']), axis=1)
    else:
        news_df['published_date_yweek'] = []

    news_df = (
        news_df
        .merge(mla_ids[['normal_name', 'PartyName', 'PartyGroup']], how='inner', on='normal_name')
        #drop the first and last weeks which could be partial
        #news_df = news_df[(news_df.published_date_week > news_df.published_date_week.min()) &
        #    (news_df.published_date_week < news_df.published_date_week.max())]
        .sort_values('published_date', ascending=False)
        .assign(
            date_pretty = lambda df: pd.to_datetime(df.published_date).dt.strftime('%Y-%m-%d'),
            title_plus_url = lambda df: df.apply(lambda row: f"{row['title']}|{row['link']}", axis=1)
        )
    )

    #Filter to most recent month
    news_sources = (
        news_df[news_df.published_date.dt.date > news_df.published_date.dt.date.max()-datetime.timedelta(days=30)]
        [['link', 'source', 'PartyGroup']]
        .drop_duplicates()
        .groupby(['source', 'PartyGroup'], as_index=False).link.count()
        .rename(index=str, columns={'link': 'News articles'})
        .sort_values('News articles', ascending=False)
        .assign(tooltip_text = lambda df: df.apply(
            lambda row: f"{row['source']}: {row['News articles']} article mention{'s' if row['News articles'] != 1 else ''} of {row['PartyGroup'].lower()}s",
            axis=1)
        )
    )
    #(now filtering to top 10/15 below in function)

    #dedup articles by party before calculating averages by party - doesn't make a big difference
    news_sentiment_by_party_week = (
        news_df[['published_date_year', 'published_date_week', 'link', 'PartyName', 'sentiment']].drop_duplicates()
        .groupby(['published_date_year', 'published_date_week', 'PartyName'], as_index=False)
        .agg({'link': len, 'sentiment': np.nanmean})
        .query("PartyName in ['DUP','Alliance','Sinn Fein','UUP','SDLP']")
    )
    #news_sentiment_by_party_week = news_sentiment_by_party_week.sort_values(['PartyName','published_date_week'], ignore_index=True)
    #news_sentiment_by_party_week = news_sentiment_by_party_week[news_sentiment_by_party_week.sentiment.notnull()]
    #news_sentiment_by_party_week = news_sentiment_by_party_week[news_sentiment_by_party_week.url >= 3]  #OK to keep in now because using smoothing
    
    #fill in missing weeks before averaging - works for volume only
    uniques = news_sentiment_by_party_week[['published_date_year', 'published_date_week']].drop_duplicates()
    uniques = pd.concat([uniques.assign(PartyName = p) for p in news_sentiment_by_party_week.PartyName.unique()]).reset_index(drop=True)
    news_sentiment_by_party_week = uniques.merge(news_sentiment_by_party_week, on=['published_date_year','published_date_week','PartyName'], how='left')
    #keep missing sentiment weeks as NA but can fill volumes as zero
    news_sentiment_by_party_week['link'] = news_sentiment_by_party_week['link'].fillna(0)

    news_sentiment_by_party_week = news_sentiment_by_party_week.join(
        news_sentiment_by_party_week.groupby('PartyName', sort=False).link\
            .rolling(news_volume_average_window_weeks, min_periods=1, center=True).mean().reset_index(0),  #the 0 is vital here
        rsuffix='_smooth').rename(index=str, columns={'link_smooth':'vol_smooth'})
    #print(news_sentiment_by_party_week.head())
    #drop first and last weeks here instead (as they are incomplete), so that table still shows the most recent articles
    news_sentiment_by_party_week['yearweekcomb'] = news_sentiment_by_party_week['published_date_year'] + news_sentiment_by_party_week['published_date_week']
    print('TODO drop first, last news weeks')
    #news_sentiment_by_party_week = news_sentiment_by_party_week[(news_sentiment_by_party_week.yearweekcomb > news_sentiment_by_party_week.yearweekcomb.min()) &
    #    (news_sentiment_by_party_week.yearweekcomb < news_sentiment_by_party_week.yearweekcomb.max())]
    #This is used for the boxplot - need mean value by party for tooltip
    news_sentiment_by_party_week = news_sentiment_by_party_week.merge(
        news_sentiment_by_party_week.groupby('PartyName').agg( 
            tooltip_text = pd.NamedAgg('sentiment', lambda s: f"Mean sentiment score = {np.mean(s):.3f}")
        ), on='PartyName', how='inner')

    return news_df, news_sources, news_sentiment_by_party_week

def load_news_summaries(data_dir, config, mla_ids):
    """ """
    # Read from file, remove rows with missing news_coverage_summary, take the latest create_date per normal_name
    news_summaries = (
        feather.read_dataframe(data_dir + config['DATA_NEWS_SUMMARIES'])
        #.dropna(subset=['news_coverage_summary'])
        .sort_values('create_date')
        .drop_duplicates(subset='normal_name', keep='last')
        .sort_values('normal_name')
        # Keep active politicians only - filter on mla_ids
        .merge(mla_ids[['normal_name']], on='normal_name', how='right')
        .fillna({'news_coverage_summary': 'NONE'})
    )
    return news_summaries

# Polls
def load_elections(data_dir, config):
    """ """
    elections_df = (
        pd.merge(
            pd.read_csv(data_dir + config['DATA_ELECTION_IDS']),
            pd.read_csv(data_dir + config['DATA_ELECTION_RESULTS']),
            on='election_id', how='inner'
        )
        .assign(date = lambda df: pd.to_datetime(df.date))
        .sort_values(['date', 'party'], ascending=[False, True])
        .assign(date_year = lambda df: df.date.dt.strftime('%Y'))
    )
    elections_df = elections_df[elections_df.date > pd.to_datetime('2015-01-01')]
    if len(elections_df) > 0:
        elections_df['tooltip_text'] = elections_df.apply(
            lambda row: f"{row['party']}: {row['pct']:g}% (election; {row['date_year']} {row['election_type']})", axis=1
        )
    else:
        elections_df['tooltip_text'] = []

    return elections_df

def load_polls(data_dir, config):
    """ """
    polls_df = (
        pd.merge(
            pd.read_csv(data_dir + config['DATA_POLL_IDS']),
            pd.read_csv(data_dir + config['DATA_POLL_RESULTS']),
            on='poll_id', how='inner'
        )
        .query("party != 'Other'")  #no point including Other
        .assign(date = lambda df: pd.to_datetime(df.date))
        .sort_values(['date', 'party'], ascending=[False, True])
        .assign(date_pretty = lambda df: df.date.dt.strftime('%Y-%m-%d'))
    )
    polls_df = polls_df[polls_df.date > pd.to_datetime('2015-01-01')]
    if len(polls_df) > 0:
        polls_df['tooltip_text'] = polls_df.apply(
            lambda row: f"{row['party']}: {row['pct']:g}% (poll; {row['organisation']}, n={row['sample_size']:.0f})", axis=1
        )
    else:
        polls_df['tooltip_text'] = []

    return polls_df

def make_poll_avgs_track(polls_df, elections_df, test_mode=False):
    """ """
    #Calculate poll averages - now uses both polls and elections
    poll_track_timestep = 100 if test_mode else 10
    earliest_poll_or_election_date = min(polls_df.date.min(), elections_df.date.min())
    poll_track_date_range = [datetime.datetime.today() - pd.to_timedelta(i, unit='day') \
        for i in range(0, (datetime.datetime.today() - earliest_poll_or_election_date).days+50, poll_track_timestep)]
    poll_avgs_track = []
    for party in polls_df[polls_df.party != 'Other'].party.unique():
        poll_avgs_track.append(
            pd.DataFrame({
                'party': party,
                'date': poll_track_date_range,
                'pred_pct': [get_current_avg_poll_pct(polls_df, elections_df, party, d) for d in poll_track_date_range]
            })
        )
    poll_avgs_track = pd.concat(poll_avgs_track)
    return poll_avgs_track

def load_election_forecast(data_dir, elct_files_date_string):
    """ Load the election forecast files (Assembly election 2022 forecast) """
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

    #Describe the possible party seat outcomes in words
    elct_fcst_cands_summary['party_seats_string'] = elct_fcst_cands_summary.apply(make_prob_string, axis=1)
    #Pretty print party pct changes from last time
    elct_fcst_cands_summary['delta_party_fp_pct_pprint'] = elct_fcst_cands_summary.delta_party_fp_pct.apply(lambda p: f"{p:+.1f}%")
    elct_fcst_cands_summary['delta_party_fp_pct_pprint'][
        (elct_fcst_cands_summary.delta_party_fp_pct == elct_fcst_cands_summary.party_mean_fp_pct) | pd.isnull(elct_fcst_cands_summary.delta_party_fp_pct)] = 'n/a'

    return (
        elct_fcst_cw_fps,
        elct_fcst_ens_party_seats,
        elct_fcst_cands_summary,
        elct_fcst_seat_deltas,
        biggest_party_fracs
    )

def get_member_mla_ids_stuff(
    mla_ids_row,
    member_other_photo_links,
    mp_api_numbers,
    mla_minister_roles,
    committee_roles,
    committee_meeting_attendance,
    mla_interests,
    allpartygroup_memberships
    ):
    """ """
    person_choice = mla_ids_row.normal_name
    print(mla_ids_row)
    print(person_choice)

    date_added = mla_ids_row['added']

    person_name_string = f"{person_choice}"
    if mla_ids_row['role'] in ['MLA', 'MP']:
        person_name_string = person_name_string + f" {mla_ids_row['role']}"

    person_constit = mla_ids_row['ConstituencyName']

    person_is_mla = mla_ids_row['role'] == 'MLA'
    
    image_url = find_profile_photo(person_is_mla, mla_ids_row.PersonId, person_choice, member_other_photo_links,
        mp_api_number=mp_api_numbers.get(person_choice, None))
    
    email_address = mla_ids_row.AssemblyEmail if person_is_mla else mla_ids_row.WestminsterEmail
    if email_address == 'none':
        email_address = None

    if person_choice in mla_minister_roles.keys():
        person_name_string += f"</br>({mla_minister_roles[person_choice]})"

    if person_is_mla and mla_ids_row['normal_name'] in committee_roles.normal_name.tolist() and mla_ids_row['normal_name'] in committee_meeting_attendance.normal_name.tolist():
        person_committee_roles_raw = committee_roles[committee_roles.normal_name == mla_ids_row['normal_name']].Organisation.tolist()
        person_committee_roles = committee_roles[committee_roles.normal_name == mla_ids_row['normal_name']].apply(
            lambda row: f"{row['Organisation']}{ ' ('+row['Role']+')' if 'Chair' in row['Role'] else ''}", axis=1
        ).tolist()

        person_committee_attendances = (
            committee_meeting_attendance[committee_meeting_attendance.normal_name == mla_ids_row['normal_name']]
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

    if person_is_mla and mla_ids_row['normal_name'] in mla_interests.normal_name.tolist():
        person_interests = mla_interests[mla_interests.normal_name==mla_ids_row['normal_name']].apply(
            lambda row: f"{row.RegisterCategory}: {row.RegisterEntry}", axis=1
        ).tolist()
    else:
        person_interests = ["No interests declared."]

    if person_is_mla and mla_ids_row['normal_name'] in allpartygroup_memberships.normal_name.tolist():
        person_apgs = (allpartygroup_memberships[allpartygroup_memberships.normal_name==mla_ids_row['normal_name']]
            .apg_name.unique().tolist()
            )
    else:
        person_apgs = []

    return (
        date_added, person_name_string, person_constit, person_is_mla,
        image_url, email_address, person_committee_work, person_interests, person_apgs
    )

def get_member_news_stuff(mla_ids_row, news_df, news_summaries):
    """ """
    person_choice = mla_ids_row.normal_name

    news_articles_last_month = news_df[(news_df.normal_name==mla_ids_row['normal_name']) &
        (news_df.published_date.dt.date > datetime.date.today()-datetime.timedelta(days=30))].shape[0]

    tmp = news_df[news_df.normal_name==mla_ids_row['normal_name']].published_date_yweek.value_counts()
    tmp = pd.DataFrame({'published_date_yweek': tmp.index, 'n_mentions': tmp.values})
    news_articles_by_week = (
        news_df[['published_date_yweek']].drop_duplicates()
        .merge(tmp, on='published_date_yweek', how='left')
        .fillna(0).sort_values('published_date_yweek')['n_mentions'].astype(int).to_list()
    )
    print('TODO change this code in get_member_news_stuff')

    if person_choice in news_summaries.normal_name.tolist():
        member_news_summaries = news_summaries[news_summaries.normal_name==person_choice].iloc[0]
        if member_news_summaries.n_articles > 0:
            news_summary_desc_string = (
                f"(Summary of {int(member_news_summaries.n_articles)} articles "
                f"in the period from {member_news_summaries.time_period.split('_')[0]} to now)"
                )
        else:
            news_summary_desc_string = ""
        news_summary_summary = member_news_summaries.news_coverage_summary.replace('\n', '<br/>')
    else:
        # members that have never had a news article will not be in the summaries table at all
        news_summary_desc_string, news_summary_summary = "", ""

    return news_articles_last_month, news_articles_by_week, news_summary_desc_string, news_summary_summary

def get_member_assembly_stuff(mla_ids_row, votes_df, questions_df, scored_plenary_contribs_df, emotions_df, n_active_mlas):
    """ """
    mla_votes_list = (
        votes_df.loc[votes_df['normal_name'] == mla_ids_row['normal_name']]
        .sort_values('DivisionDate', ascending=False)
        [['DivisionDate', 'motion_plus_url', 'Vote']]
    )
    mla_votes_list = [r[1].values.tolist() for r in mla_votes_list.iterrows()]

    #Account for people joining midway through a session
    # TODO check
    vote_date_added = '2020-01-11' if mla_ids_row['added'] == '2020-08-01' else mla_ids_row['added'] #I tracked most people from 1 Aug 2020 but have their vote info from Jan 2020
    votes_they_joined_before = votes_df[(votes_df.DivisionDate.astype(str) >= vote_date_added) | (votes_df.normal_name==mla_ids_row['normal_name'])]
    votes_present_numbers = (sum(votes_df['normal_name'] == mla_ids_row['normal_name']), votes_they_joined_before.EventId.nunique())
    if len(votes_they_joined_before) > 0:
        votes_present_first_date = votes_they_joined_before.DivisionDate.min().strftime('%d %B %Y')
    else:
        votes_present_first_date = 'joining the Assembly'
    votes_present_string = f"<b>{votes_present_numbers[0]} / {votes_present_numbers[1]} votes</b> since {votes_present_first_date}"

    num_questions = (questions_df['normal_name'] == mla_ids_row['normal_name']).sum()
    member_question_volumes = questions_df['normal_name'].value_counts().values.tolist()
    if num_questions > 0:
        questions_rank = questions_df['normal_name'].value_counts().index.get_loc(mla_ids_row['normal_name']) + 1
    else:
        questions_rank = max(n_active_mlas, questions_df['normal_name'].nunique())
    questions_rank_string = f"<b>#{questions_rank} / {max(n_active_mlas, questions_df['normal_name'].nunique())}</b>"
    #can't consistently work out the denominator excluding ministers so just use the 90

    num_plenary_contribs = (scored_plenary_contribs_df['speaker'] == mla_ids_row['normal_name']).sum()
    if num_plenary_contribs > 0:
        plenary_contribs_rank = scored_plenary_contribs_df['speaker'].value_counts().index.get_loc(mla_ids_row['normal_name']) + 1
    else:
        plenary_contribs_rank = max(n_active_mlas, scored_plenary_contribs_df.speaker.nunique())
    plenary_contribs_rank_string = f"<b>#{plenary_contribs_rank} / {max(n_active_mlas, scored_plenary_contribs_df.speaker.nunique())}</b>"
    member_contribs_volumes = scored_plenary_contribs_df['speaker'].value_counts().values.tolist()

    top_contrib_topics = (
        scored_plenary_contribs_df[scored_plenary_contribs_df['speaker'] == mla_ids_row['normal_name']]
        .topic_name.value_counts(normalize=True, dropna=False)
    )
    top_contrib_topics = top_contrib_topics[top_contrib_topics.index != 'misc./none']
    #send topic|pct for font size|color for font
    if len(top_contrib_topics) >= 3:
        plenary_contribs_colour_dict = get_plenary_contribs_colour_dict()
        top_contrib_topic_list = [f"{top_contrib_topics.index[0]} ({top_contrib_topics.values[0]*100:.0f}%)|{36*max(min(top_contrib_topics.values[0]/0.4,1), 16/36):.0f}|{plenary_contribs_colour_dict[top_contrib_topics.index[0]]}",
            f"{top_contrib_topics.index[1]} ({top_contrib_topics.values[1]*100:.0f}%)|{36*max(min(top_contrib_topics.values[1]/0.4,1), 16/36):.0f}|{plenary_contribs_colour_dict[top_contrib_topics.index[1]]}",
            f"{top_contrib_topics.index[2]} ({top_contrib_topics.values[2]*100:.0f}%)|{36*max(min(top_contrib_topics.values[2]/0.4,1), 16/36):.0f}|{plenary_contribs_colour_dict[top_contrib_topics.index[2]]}"
        ]
    else:
        top_contrib_topic_list = []
    #emotions
    member_emotion_ranks_string = "Plenary contributions language scores relatively <b>high</b> on "
    member_any_top_emotion = False
    for emotion_type in ['anger', 'anticipation', 'joy', 'sadness', 'trust']:
        tmp = emotions_df[(emotions_df.emotion_type==emotion_type) & (emotions_df.word_count >= 100)].sort_values('ave_emotion', ascending=False)
        if mla_ids_row['normal_name'] in tmp.speaker.tolist():
            member_rank = (tmp.speaker==mla_ids_row['normal_name']).idxmax()+1
            if member_rank <= 15:
                member_any_top_emotion = True
                member_emotion_ranks_string += f"<b>{emotion_type}</b> (#{member_rank}/{tmp.shape[0]}), "
    if member_any_top_emotion:
        member_emotion_ranks_string = member_emotion_ranks_string[:-2]
    else:
        member_emotion_ranks_string = None
        #bit of a simplification because words can score for two emotions, but roughly
        #  estimate total emotion by adding 5 core emotion fractions
        tmp = (
            emotions_df[
                (emotions_df.emotion_type.isin(['anger', 'anticipation', 'joy', 'sadness', 'trust', 'disgust', 'fear', 'surprise'])) &
                (emotions_df.word_count >= 100)
            ]
            .groupby('speaker', as_index=False)
            .agg({'ave_emotion': sum})
            .sort_values('ave_emotion', ascending=True)
        )
        if mla_ids_row['normal_name'] in tmp.speaker.tolist():
            member_rank = (tmp.speaker==mla_ids_row['normal_name']).idxmax()+1
            if member_rank <= 20:
                member_emotion_ranks_string = f"Plenary contributions language scores relatively <b>low on emotion</b> overall (<b>#{tmp.shape[0]-member_rank+1} / {tmp.shape[0]}</b>)"
    
    return (
        mla_votes_list, votes_present_string, votes_present_numbers,
        num_questions, questions_rank_string, member_question_volumes,
        num_plenary_contribs, plenary_contribs_rank_string, member_contribs_volumes,
        top_contrib_topic_list, member_emotion_ranks_string
    )

def get_member_sm_stuff(mla_ids_row, sm_posts_df, member_reposts, member_post_sentiment):
    """ """
    # Limit all the sm bits to the last year
    sm_posts_last_year = sm_posts_df[sm_posts_df.created_at_dt.dt.date > datetime.date.today()-datetime.timedelta(days=365)]

    sm_n_active_people = sm_posts_last_year.normal_name.nunique()
    sm_active_people_rank_split_points = [x*sm_n_active_people for x in [0.2, 0.4, 0.7]]
    # these must have 1 more value than split points:
    sm_post_volume_rank_strings = [
        "Posts on Bluesky <b>frequently</b>",
        "Posts on Bluesky <b>fairly frequently</b>",
        "Posts on Bluesky at an <b>average rate</b>",
        "<b>Doesn't post on Bluesky very often</b>",
    ]
    sm_reposts_n_active_people = member_reposts.normal_name.nunique()
    sm_reposts_active_people_rank_split_points = [x*sm_reposts_n_active_people for x in [0.2, 0.4, 0.7]]
    repost_rate_rank_strings = [
        "<b>High</b> Bluesky impact",
        "<b>Fairly high</b> Bluesky impact",
        "<b>Average</b> Bluesky impact",
        "<b>Low</b> Bluesky impact",
    ]
    sm_sentiment_n_active_people = member_post_sentiment.normal_name.nunique()
    sm_sentiment_active_people_rank_split_points = [x*sm_sentiment_n_active_people for x in [0.2, 0.4, 0.7]]
    sm_post_positivity_rank_strings = [
        "Posts <b>very positive</b> messages",
        "Posts <b>fairly positive</b> messages",
        "Posts messages of <b>average sentiment</b>",
        "Posts <b>relatively negative</b> messages",
    ]

    #Better to do last month as I will only be updating weekly
    n_sm_posts_last_month = sm_posts_last_year[
        (sm_posts_last_year.normal_name==mla_ids_row['normal_name']) &
        (sm_posts_last_year.created_at_dt.dt.date > datetime.date.today()-datetime.timedelta(days=30))
    ].shape[0]


    sm_posts_week_range = [
        f'{x[1].year}-{x[1].week:02g}' 
        for x in pd.date_range(start=sm_posts_last_year.created_at_dt.min(), end=sm_posts_last_year.created_at_dt.max(), freq='W').isocalendar().iterrows()
    ]    
    sm_posts_by_week = (
        pd.DataFrame({'created_at_yweek': sm_posts_week_range})
        .merge(
            sm_posts_last_year[sm_posts_last_year.normal_name==mla_ids_row['normal_name']]
            .groupby('created_at_yweek', as_index=False)
            .agg(n_posts = ('status_id', len)),
            on='created_at_yweek', how='left')
        .fillna(0)
        .sort_values('created_at_yweek')
        ['n_posts'].astype(int).to_list()
    )
    # this is counts for this person by week from the first week recorded in sm_posts_last_year, for the indiv sparkline

    member_sm_post_volumes = sm_posts_last_year['normal_name'].value_counts().values.tolist()
    member_repost_rates = member_reposts['reposts_per_post'].tolist()
    member_sm_post_positivities = member_post_sentiment['sentiment_vader_compound'].tolist()

    sm_handle, repost_rate, sm_post_positivity = None, None, None
    sm_post_volume_rank_string = "We don't know of a Bluesky account for this member"
    repost_rate_rank_string, sm_post_positivity_rank_string = 'n/a', 'n/a'
    sample_recent_posts = pd.DataFrame()

    if sum(sm_posts_last_year.normal_name==mla_ids_row['normal_name']) > 0:

        sm_post_volume_rank = sm_posts_last_year['normal_name'].value_counts().index.get_loc(mla_ids_row['normal_name']) + 1
        sm_post_volume_rank_string = (
            sm_post_volume_rank_strings[get_rank_category_index(sm_post_volume_rank, sm_active_people_rank_split_points)] +
            f"<br />(<b>#{sm_post_volume_rank} / {sm_n_active_people}</b> in total posts among those active)"
        )

        #member_reposts requires having at least 10 posts in last 5 weeks
        if mla_ids_row['normal_name'] in member_reposts['normal_name'].tolist():
            repost_rate = member_reposts[member_reposts['normal_name'] == mla_ids_row['normal_name']]['reposts_per_post'].iloc[0]
            repost_rate_rank = (member_reposts['normal_name'] == mla_ids_row['normal_name']).values.argmax() + 1
            # ranks are from a smaller list
            repost_rate_rank_string = (
                repost_rate_rank_strings[get_rank_category_index(repost_rate_rank, sm_reposts_active_people_rank_split_points)] +
                f"<br />(<b>#{repost_rate_rank} / {sm_reposts_n_active_people}</b> in reposts per original post)"
            )
        
        if mla_ids_row['normal_name'] in member_post_sentiment.normal_name.tolist():
            sm_post_positivity = member_post_sentiment[member_post_sentiment['normal_name'] == mla_ids_row['normal_name']]['sentiment_vader_compound'].iloc[0]
            sm_post_positivity_rank = (member_post_sentiment['normal_name'] == mla_ids_row['normal_name']).values.argmax()+1
            # ranks are from a smaller list
            sm_post_positivity_rank_string = (
                sm_post_positivity_rank_strings[get_rank_category_index(sm_post_positivity_rank, sm_sentiment_active_people_rank_split_points)] +
                f"<br />(<b>#{sm_post_positivity_rank} / {sm_sentiment_n_active_people}</b> for social media positivity)"
            )

        sample_recent_posts = (
            sm_posts_last_year[
                (sm_posts_last_year.normal_name==mla_ids_row['normal_name']) &
                (~sm_posts_last_year.is_repost)
            ]
            .sort_values('created_at_dt', ascending=False)
            .head(15)
            [['created_at_dt', 'text', 'involves_quote']]
        )
        if len(sample_recent_posts) > 0:
            sample_recent_posts = (
                sample_recent_posts
                .assign(
                    created_at = lambda df: df.created_at_dt.dt.strftime('%Y-%m-%d'),
                    quoted_url = lambda df: df.apply(
                        lambda row: re.findall('//t.*', row['text'])[0] if row['involves_quote'] else '', axis=1),
                    text = lambda df: df.text.str.replace('//t.*', '', regex=True)
                )
                .query("text != ''")
                .sample(frac=1.0).head(5).sort_values('created_at', ascending=False) # limit to 5
            )
    elif sum(sm_posts_df.normal_name==mla_ids_row['normal_name']) > 0:
        # They do appear in sm_posts but not in the last year
        sm_post_volume_rank_string = "Not active on Bluesky in the past year"

    return (
        sm_handle, n_sm_posts_last_month, sm_posts_by_week,
        sm_post_volume_rank_string, member_sm_post_volumes,
        repost_rate_rank_string, repost_rate, member_repost_rates,
        sm_post_positivity_rank_string, sm_post_positivity, member_sm_post_positivities,
        sample_recent_posts
    )
