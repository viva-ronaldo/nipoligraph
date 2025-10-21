from flask import Flask, render_template, url_for, \
    request, flash, jsonify, redirect, send_from_directory
import requests
import datetime, re, getpass, os, time
import feather, json, pickle, yaml
import pandas as pd
import altair as alt
import numpy as np
from itertools import product

from nipol_etl_functions import (
    get_plenary_contribs_colour_dict,
    get_parties_colour_dict,
    assign_assembly_session_name,
    get_current_avg_poll_pct,
    make_prob_string,
    find_profile_photo,
    load_blog_item_table,
    get_rank_category_index,
    #
    load_member_ids,
    load_minister_roles,
    load_and_process_assembly_committees,
    load_mla_interests,
    load_allpartygroup_memberships,
    load_assembly_diary,    
    load_and_process_twitter,
    load_and_process_bluesky,
    load_and_process_assembly_questions,
    load_and_process_assembly_answers,
    load_and_process_assembly_votes,
    load_and_process_assembly_plenary_contribs,
    load_and_process_assembly_contrib_emotions,
    load_and_process_all_historical_assembly,
    load_and_process_news_data,
    load_news_summaries,
    load_elections,
    load_polls,
    make_poll_avgs_track,
    load_election_forecast,
    get_member_mla_ids_stuff,
    get_member_news_stuff,
    get_member_assembly_stuff,
    get_member_sm_stuff,
)
from nipol_plot_functions import (
    add_grey_legend,
    plot_user_sm_post_num_fn,
    plot_user_repost_fn,
    plot_user_sm_post_sentiment_fn,
    #plot_tweet_pca_all_mlas_fn,
    polls_plot_fn,
    plot_questions_asked_fn,
    plot_minister_answer_times_fn,
    plot_vote_pca_all_mlas_fn,
    plot_plenary_topics_overall_fn,
    plot_plenary_emotions_fn,
    plot_news_sources_fn,
    shared_plot_news_fn,
    plot_constituency_depriv_metrics_fn,
    elct_cw_bars_fn,
    elct_cw_seats_range_fn,
    elct_cw_delta_seats_map_fn,
    elct_cw_most_seats_fn,
)

if getpass.getuser() == 'david':
    data_dir = '/media/shared_storage/data/nipol_aws_copy/data/'
    config_dir = '../'
    test_mode = True
else:
    data_dir = '/home/vivaronaldo/nipol/data/'
    config_dir = '/home/vivaronaldo/nipol/'
    test_mode = False

with open(config_dir + 'config.yaml', 'r') as f:
    config = yaml.safe_load(f)

app = Flask(__name__)
app.debug = False
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

#Assembly page sessions
CURRENT_ASSEMBLY_SESSION = config['CURRENT_ASSEMBLY_SESSION']

#Election forecast is usually off
show_election_forecast = config['INCLUDE_ELECTION_FORECAST']
# From mid-2023, can't scrape tweets, so exclude from indiv page and note on twitter page
include_twitter = config['INCLUDE_TWITTER']
include_bluesky = config['INCLUDE_BLUESKY']

#Read member data
#-----------
plenary_contribs_colour_dict = get_plenary_contribs_colour_dict()

mla_ids, party_names_translation, party_names_translation_long = load_member_ids(
    data_dir, config)
mla_ids_alltime, _, _ = load_member_ids(data_dir, config, active_only=False)

#Numbers to find MP portraits on parliament website
with open(data_dir + 'mp_api_numbers.json', 'r') as f:
    mp_api_numbers = json.load(f)
with open(data_dir + 'member_other_photo_links.json', 'r') as f:
    member_other_photo_links = json.load(f)

party_colours = get_parties_colour_dict(data_dir, config)

mla_minister_roles = load_minister_roles(data_dir, config, mla_ids)

print('Done ids')

#Social media
#-------
if include_twitter:
    sm_posts_df, top_posters, member_reposts, member_post_sentiment = load_and_process_twitter(
        data_dir, config, mla_ids)

    tweets_network_top5s = pd.read_csv(data_dir + 'tweets_network_last3months_top5s.csv')

    #Generated tweets
    gen_tweets = pd.read_csv(data_dir + 'fiveparties_generated_tweets_1epoch_gpt2medium_reppen1pt5.csv')
    #exclude any that are all hashtags/mentions
    gen_tweets['all_h_or_m'] = gen_tweets.generated_text.apply(lambda t: all([w[0] in ['@','#'] for w in t.split()]))
    gen_tweets = gen_tweets[~gen_tweets.all_h_or_m]

    print('Done Twitter')

elif include_bluesky:
    sm_posts_df, top_posters, member_reposts, member_post_sentiment = load_and_process_bluesky(
        data_dir, config, mla_ids)

    print('Done Bluesky')

#Assembly 
#--------
questions_df, questioners = load_and_process_assembly_questions(data_dir, config, mla_ids_alltime)
answers_df, minister_time_to_answer = load_and_process_assembly_answers(
    data_dir, config, mla_ids_alltime, mla_minister_roles)
print('Done questions and answers')

#Votes
votes_df, vote_results_df, mla_votes_pca, mlas_2d_rep, v_comms = load_and_process_assembly_votes(
    data_dir, config, mla_ids)
scored_plenary_contribs_df, lda_top5s, plenary_contribs_topic_counts = load_and_process_assembly_plenary_contribs(
    data_dir, config)
emotions_df, emotions_party_agg = load_and_process_assembly_contrib_emotions(
    data_dir, config, mla_ids)
print('Done votes and contributions')

# Other Assembly
diary_df = load_assembly_diary(data_dir, config)
committee_roles, committee_meeting_attendance, committee_agendas_agg = load_and_process_assembly_committees(
    data_dir, config, mla_ids)
mla_interests = load_mla_interests(data_dir, config, mla_ids)
allpartygroup_memberships = load_allpartygroup_memberships(data_dir, config, mla_ids)
print('Done other Assembly')

#Assembly historical
#-------------------
#Read in original hist files for 2007-2019 sessions, and append
#  later sessions that become historical, which is 2020-2022
(
    hist_mla_ids,
    hist_questions_df, hist_questioners, hist_answers_df, hist_minister_time_to_answer,
    hist_votes_df, hist_votes_pca_res, hist_v_comms,
    hist_plenary_contribs_df, hist_plenary_contribs_topic_counts, hist_emotions_party_agg
) = load_and_process_all_historical_assembly(data_dir, config, lda_top5s)
print('Done historical contributions')

#News 
#----
news_df, news_sources, news_sentiment_by_party_week = load_and_process_news_data(
    data_dir, config, mla_ids, news_volume_average_window_weeks=config['NEWS_VOLUME_PLOT_AVERAGE_WINDOW_WEEKS'])
news_summaries = load_news_summaries(data_dir, config, mla_ids)
print('Done news')

#Polls 
#-----
polls_df = load_polls(data_dir, config)
elections_df = load_elections(data_dir, config)
poll_avgs_track = make_poll_avgs_track(polls_df, elections_df, test_mode=test_mode)
print('Done polls')

#Election forecast
#-----------------
if show_election_forecast:
    elct_files_date_string = config['ELECTION_FORECAST_VERSION']
    (
        elct_fcst_cw_fps,
        elct_fcst_ens_party_seats,
        elct_fcst_cands_summary,
        elct_fcst_seat_deltas,
        biggest_party_fracs
    ) = load_election_forecast(data_dir, elct_files_date_string)

#Other
#-----
#Blog articles list
blog_pieces = pd.read_csv(data_dir + 'blog_pieces_list.psv', sep='|').iloc[-1::-1]
blog_pieces = blog_pieces[blog_pieces.date_added <= datetime.date.today().strftime('%Y-%m-%d')]
print('Done blog')

#Postcode stuff
postcodes_to_constits = pd.read_csv(data_dir + config['DATA_POSTCODES_CONSTITS'], index_col=None)
combined_demog_table = pd.read_csv(data_dir + config['DATA_CONSTITS_DEMOGRAPHICS'])
#These are in increasing order, i.e. lowest age gets age_rank_order=1
postcode_and_constits_list = sorted(postcodes_to_constits.Postcode.tolist() + mla_ids.ConstituencyName.unique().tolist())
sorted_mlas_mps_list = sorted(mla_ids.normal_name.tolist())
print('Done postcodes')

#Totals for front page
#---------------------
n_politicians_current_session = len(mla_ids)
rank_split_points = [10, n_politicians_current_session*0.3, n_politicians_current_session*0.7]

n_active_mlas = (mla_ids.role=='MLA').sum()
#n_active_mlas_excl_ministers = n_active_mlas - len(mla_minister_roles.keys())

file_change_times = [os.path.getmtime(x) for x in \
    [data_dir + config['DATA_BLUESKY_POSTS'],
     data_dir + config['DATA_ASSEMBLY_DIARY_EVENTS'],
     data_dir + config['DATA_NEWS_WORLDNEWS']]
]
last_updated_date = time.strftime('%A, %-d %B', time.localtime(max(file_change_times)))

totals_dict = {
    'n_politicians': f"{pd.concat([mla_ids[['normal_name']], hist_mla_ids[['normal_name']]]).normal_name.nunique():,}",
    'n_questions': f"{pd.concat([questions_df, hist_questions_df]).DocumentId.nunique():,}",
    'n_answers': f"{pd.concat([answers_df, hist_answers_df]).DocumentId.nunique():,}",
    'n_votes': f"{pd.concat([votes_df, hist_votes_df]).EventId.nunique():,}",
    'n_contributions': f"{pd.concat([scored_plenary_contribs_df, hist_plenary_contribs_df]).shape[0]:,}",
    'n_posts': f"{sm_posts_df.status_id.nunique():,}",
    'n_news': f"{news_df.link.nunique():,}",
    'n_polls': f"{polls_df.poll_id.nunique():,}",
    'last_updated_date': last_updated_date
}

#Tidy up
del answers_df, hist_answers_df, hist_questions_df, vote_results_df, hist_plenary_contribs_df

# ----
# App routes

def create_route(endpoint, func, *args, **kwargs):
    """ Determine a unique name for the endpoint based on the route """
    def wrapper():
        return func(*args, **kwargs)
    # Use the endpoint parameter as the name for the view function
    wrapper.__name__ = endpoint.strip('/').replace('/', '_')  # Generate a unique function name
    app.add_url_rule(endpoint, view_func=wrapper)  # Use add_url_rule to register the URL with the custom name

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
        full_mla_list = sorted_mlas_mps_list,
        postcodes_list = postcode_and_constits_list,
        blog_pieces = blog_pieces[:3])

@app.route('/what-they-say', methods=['GET'])
def social_media():
    if include_bluesky:
        return render_template('bluesky.html',
            full_mla_list = sorted_mlas_mps_list,
            postcodes_list = postcode_and_constits_list)
    elif include_twitter:
        return render_template('twitter.html',
            full_mla_list = sorted_mlas_mps_list,
            postcodes_list = postcode_and_constits_list,
            info_centr_top5 = tweets_network_top5s['info_centr'].tolist(),
            page_rank_top5 = tweets_network_top5s['page_rank'].tolist(),
            betw_centr_top5 = tweets_network_top5s['betw_centr'].tolist(),
            gen_tweets_dup = gen_tweets.loc[gen_tweets.mla_party=='DUP', 'generated_text'].sample(100).tolist(),
            gen_tweets_uup = gen_tweets.loc[gen_tweets.mla_party=='UUP', 'generated_text'].sample(100).tolist(),
            gen_tweets_alli = gen_tweets.loc[gen_tweets.mla_party=='Alliance', 'generated_text'].sample(100).tolist(),
            gen_tweets_sdlp = gen_tweets.loc[gen_tweets.mla_party=='SDLP', 'generated_text'].sample(100).tolist(),
            gen_tweets_sf = gen_tweets.loc[gen_tweets.mla_party=='Sinn Fein', 'generated_text'].sample(100).tolist())
    else:
        return render_template('index.html',
            totals_dict = totals_dict,
            full_mla_list = sorted_mlas_mps_list,
            postcodes_list = postcode_and_constits_list,
            blog_pieces = blog_pieces[:3])

@app.route('/twitter')
def twitter():
    return redirect(url_for('social_media'))

@app.route('/what-they-do', methods=['GET'])
def assembly():
    args = request.args
    if 'assembly_session' in args:
        session_to_plot = args.get('assembly_session')
    else:
        session_to_plot = CURRENT_ASSEMBLY_SESSION

    #Exclude events that have now happened
    #But not more than 6 items, so as not to clutter page
    diary_df_filtered = (
        diary_df[diary_df['EndTime'] >= datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')]
        .head(6)
    )

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

    chances_to_take_a_side = v_comms_tmp[(v_comms_tmp.uni_nat_split=='Yes')].shape[0]
    alli_num_votes_with_uni = v_comms_tmp[(v_comms_tmp.uni_nat_split=='Yes') &
        (v_comms_tmp.alli_vote == v_comms_tmp.uni_bloc_vote) &
        (v_comms_tmp.alli_vote != 'ABSTAINED')].shape[0]
    alli_num_votes_with_nat = v_comms_tmp[(v_comms_tmp.uni_nat_split=='Yes') & 
        (v_comms_tmp.alli_vote == v_comms_tmp.nat_bloc_vote) &
        (v_comms_tmp.alli_vote != 'ABSTAINED')].shape[0]
    green_num_votes_with_uni = v_comms_tmp[(v_comms_tmp.uni_nat_split=='Yes') &
        (v_comms_tmp.green_vote == v_comms_tmp.uni_bloc_vote) &
        (v_comms_tmp.green_vote != 'ABSTAINED')].shape[0]
    green_num_votes_with_nat = v_comms_tmp[(v_comms_tmp.uni_nat_split=='Yes') & 
        (v_comms_tmp.green_vote == v_comms_tmp.nat_bloc_vote) &
        (v_comms_tmp.green_vote != 'ABSTAINED')].shape[0]
    # Remove the green vote part if there are no Green MLAs
    # (in assembly.html, identify this with green_like_uni_details[0] + green_like_nat_details[0] == 0)
    cols_for_votes_list = [
        'vote_date', 'vote_subject', 'vote_tabler_group', 'vote_result',
        'uni_bloc_vote', 'nat_bloc_vote',
        'alli_vote', 'green_vote', 'uni_nat_split'
    ]
    if (v_comms_tmp.green_vote != 'ABSTAINED').sum() == 0:
        cols_for_votes_list = [c for c in cols_for_votes_list if c != 'green_vote']
    votes_list = [e[1].values.tolist() for e in v_comms_tmp[cols_for_votes_list].iterrows()]
    
    return render_template('assembly.html',
        session_to_plot = session_to_plot,
        current_session = CURRENT_ASSEMBLY_SESSION,
        session_names_list = config['ALL_ASSEMBLY_SESSIONS'],
        diary = diary_df_filtered, 
        n_mlas = n_mlas, 
        n_votes = n_votes,
        full_mla_list = sorted_mlas_mps_list,
        postcodes_list = postcode_and_constits_list,
        votes_list = votes_list,
        pca_votes_threshold_pct = int(config['PCA_VOTES_THRESHOLD_FRACTION']*100),
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
    news_df_dedup = (
        news_df
        .assign(surname = lambda df: df.apply(lambda row: row['normal_name'].split(' ')[-1], axis=1))
        .sort_values('surname', ascending=True)
        .groupby(['date_pretty', 'title_plus_url', 'source'], as_index=False)
        .agg(normal_names = ('normal_name', lambda l: ', '.join(l)))
        .sort_values('date_pretty', ascending=False)
    )

    n_news_articles_to_list = 50 if test_mode else 1000
    articles_list = [e[1].values.tolist() for e in news_df_dedup.head(n_news_articles_to_list)[
        ['date_pretty', 'title_plus_url', 'source', 'normal_names']].iterrows()]

    return render_template('news.html',
        articles_list = articles_list,
        full_mla_list = sorted_mlas_mps_list,
        postcodes_list = postcode_and_constits_list,
        news_volume_average_window_weeks = config['NEWS_VOLUME_PLOT_AVERAGE_WINDOW_WEEKS'],
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
    polls_tmp = polls_tmp[['date_plus_url', 'organisation', 'sample_size', 'party', 'pct']]

    if show_election_forecast:
        elct_cands_tmp = elct_fcst_cands_summary[['fullname', 'frac_elected', 'constit', 'party_short']].copy()
        elct_cands_tmp['constit'] = elct_cands_tmp.apply(
            lambda row: f"{row['constit']}|postcode?postcode_choice={row['constit'].upper().replace(' ','+')}#election", axis=1)
        elct_all_cand_list = [e[1].values.tolist() for e in elct_cands_tmp.iterrows()]
        elct_n_ensemble = elct_fcst_ens_party_seats.it.max()
    else:
        elct_all_cand_list = elct_n_ensemble = None

    return render_template('polls.html',
        poll_results_list = [e[1].values.tolist() for e in polls_tmp.iterrows()],
        full_mla_list = sorted_mlas_mps_list,
        postcodes_list = postcode_and_constits_list,
        elct_all_cand_list = elct_all_cand_list,
        elct_n_ensemble = elct_n_ensemble)

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html',
        full_mla_list = sorted_mlas_mps_list,
        postcodes_list = postcode_and_constits_list)

@app.route('/blog', methods=['GET'])
def blog():
    blog_pieces['background_image_path'] = blog_pieces.background_image_file.apply(lambda f: url_for('static', filename=f))
    return render_template('blog.html', 
        full_mla_list = sorted_mlas_mps_list,
        postcodes_list = postcode_and_constits_list,
        blog_pieces = blog_pieces)

@app.route('/blog/<post_name>', methods=['GET'])
def blog_item(post_name):
    """ """ 
    blog_links = blog_pieces['link'].tolist()
    blog_titles = blog_pieces['title'].tolist()

    if post_name in blog_links:
        place_in_blog_list = blog_links.index(post_name)

        return render_template(f'blog-{post_name}.html',
            full_mla_list = sorted_mlas_mps_list,
            postcodes_list = postcode_and_constits_list,
            prev_and_next_article_title = (
                None if place_in_blog_list==blog_pieces.shape[0]-1 else blog_titles[place_in_blog_list+1],
                None if place_in_blog_list==0 else blog_titles[place_in_blog_list-1]),
            prev_and_next_article_link = (
                None if place_in_blog_list==blog_pieces.shape[0]-1 else blog_links[place_in_blog_list+1],
                None if place_in_blog_list==0 else blog_links[place_in_blog_list-1]),
            # Pass up to 3 dfs to use as tables in the blog post; only some post_names use any tables.
            df_1_list = load_blog_item_table(data_dir, post_name, 1) if post_name in config['BLOG_ITEMS_WITH_A_TABLE_1'] else None,
            df_2_list = load_blog_item_table(data_dir, post_name, 2) if post_name in config['BLOG_ITEMS_WITH_A_TABLE_2'] else None,
            df_3_list = load_blog_item_table(data_dir, post_name, 3) if post_name in config['BLOG_ITEMS_WITH_A_TABLE_3'] else None,
            )
    else:
        return blog()        

@app.route('/individual', methods=['GET'])
def indiv():
    """ 
    """
    args = request.args
    if args.get('mla_name', 'NOT FOUND') in mla_ids.normal_name.unique():
        person_choice = args.get('mla_name')                
        person_choice_party = mla_ids[mla_ids.normal_name==person_choice].PartyName_long.iloc[0]
        person_name_lc = person_choice.lower()

        row = mla_ids[mla_ids.normal_name == person_choice].iloc[0]

        (
            date_added, person_name_string, person_constit, person_is_mla,
            image_url, email_address, person_committee_work, person_interests, person_apgs
        ) = get_member_mla_ids_stuff(
            row, member_other_photo_links, mp_api_numbers,
            mla_minister_roles, committee_roles, committee_meeting_attendance,
            mla_interests, allpartygroup_memberships
        )

        (
            sm_handle, n_sm_posts_last_month, sm_posts_by_week,
            sm_post_volume_rank_string, member_sm_post_volumes,
            repost_rate_rank_string, repost_rate, member_repost_rates,
            sm_post_positivity_rank_string, sm_post_positivity, member_sm_post_positivities,
            sample_recent_posts
        ) = get_member_sm_stuff(row, sm_posts_df, member_reposts, member_post_sentiment)

        news_articles_last_month, news_articles_by_week, news_summary_desc_string, news_summary_summary = get_member_news_stuff(
            row, news_df, news_summaries
        )

        # Assembly votes, questions, plenary
        if person_is_mla:
            (
                mla_votes_list, votes_present_string, votes_present_numbers,
                num_questions, questions_rank_string, member_question_volumes,
                num_plenary_contribs, plenary_contribs_rank_string, member_contribs_volumes,
                top_contrib_topic_list, member_emotion_ranks_string
            ) = get_member_assembly_stuff(row, votes_df, questions_df, scored_plenary_contribs_df, emotions_df, n_active_mlas)
        else:
            mla_votes_list, votes_present_string = None, None
            votes_present_numbers = [0, 0]
            num_questions, questions_rank_string, member_question_volumes = None, None, None
            num_plenary_contribs, plenary_contribs_rank_string, member_contribs_volumes = None, None, None
            top_contrib_topic_list, member_emotion_ranks_string = None, None

        return render_template('indiv.html', 
            person_is_mla = person_is_mla,
            mla_or_mp_id = row.PersonId if person_is_mla else mp_api_numbers.get(person_choice, None),
            full_mla_list = sorted_mlas_mps_list,
            postcodes_list = postcode_and_constits_list,
            person_name_string = person_name_string,
            person_name_lc = person_name_lc,
            person_date_added = date_added,
            person_party = person_choice_party,
            image_url = image_url,
            email_address = email_address,
            person_constit = person_constit,
            #
            person_committee_work = person_committee_work,
            person_interests = person_interests,
            person_apgs = person_apgs,
            #
            include_social_media = 'bluesky' if include_bluesky else ('twitter' if include_twitter else 'no'),
            sm_handle = sm_handle,
            sm_posts_last_month = n_sm_posts_last_month, 
            sm_posts_by_week = sm_posts_by_week,
            sm_post_volume_rank_string = sm_post_volume_rank_string,
            member_sm_post_volumes = member_sm_post_volumes,
            repost_rate_rank_string = repost_rate_rank_string,
            repost_rate = repost_rate,
            member_repost_rates = member_repost_rates,
            sm_post_positivity_rank_string = sm_post_positivity_rank_string,
            sm_post_positivity = sm_post_positivity,
            member_sm_post_positivities = member_sm_post_positivities,
            sample_recent_posts = sample_recent_posts,
            #
            news_tracked_since_date = 'December 2023', # if date_added < '2023-12-01' else date_added,  # confusing for MLA->MPs
            news_articles_last_month = news_articles_last_month,
            news_articles_by_week = news_articles_by_week,
            news_summary_desc_string = news_summary_desc_string,
            news_summary_summary = news_summary_summary,
            #
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
            member_emotion_ranks_string = member_emotion_ranks_string)

    else:
        blog_pieces['background_image_path'] = blog_pieces.background_image_file.apply(lambda f: url_for('static', filename=f))
        return render_template('index.html',
            totals_dict = totals_dict,
            full_mla_list = sorted_mlas_mps_list,
            postcodes_list = postcode_and_constits_list,
            blog_pieces = blog_pieces[:3])

@app.route('/postcode', methods=['GET'])
def postcode():
    args = request.args
    # if 'postcode_choice' in args:
    #     postcode_choice = args.get('postcode_choice').upper()
    # else:
    #     postcode_choice = 'BT1 1AA'
    postcode_choice = args.get('postcode_choice', 'BT1 1AA').upper()

    if postcode_choice in mla_ids.ConstituencyName.str.upper().tolist():
        # postcode_choice is actually a constit name in this case
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

    mla_choices = (
        mla_ids[(mla_ids.active == 1) & (mla_ids.ConstituencyName.str.upper() == constit_choice)]
        .sort_values(['role', 'MemberLastName'])
    )

    normal_names_list = mla_choices.normal_name.tolist()
    mla_or_mp_ids_list = mla_choices.apply(lambda row: 
        row.PersonId if row.role == 'MLA' else mp_api_numbers.get(row.normal_name, None),
        axis=1).tolist()

    rep_image_urls_list = []
    votes_present_string_list = []
    top_contrib_topic_list_list = []

    for row_tuple in mla_choices.iterrows():
        row = row_tuple[1]
        
        # image_url = find_profile_photo(row.role == 'MLA', row.PersonId, row.normal_name, member_other_photo_links,
        #     mp_api_number=mp_api_numbers.get(row.normal_name, None))
        # rep_image_urls_list.append(image_url)
        _, _, _, _, image_url, email_address, _, _, _ = get_member_mla_ids_stuff(
            row, member_other_photo_links, mp_api_numbers,
            mla_minister_roles, committee_roles, committee_meeting_attendance,
            mla_interests, allpartygroup_memberships)
        rep_image_urls_list.append(image_url)

        if row.role == 'MLA':
            _, votes_present_string, _, _, _, _, _, _, _, top_contrib_topic_list, _ = get_member_assembly_stuff(
                row, votes_df, questions_df, scored_plenary_contribs_df, emotions_df, n_active_mlas)
            # replace 'votes since date' with 'Assembly votes in the current session''
            votes_present_string = votes_present_string.split('votes</b> since')[0] + 'Assembly votes</b> in the current session'
            votes_present_string_list.append(votes_present_string)
            top_contrib_topic_list_list.append(top_contrib_topic_list)
        else:
            votes_present_string_list.append('n/a')
            top_contrib_topic_list_list.append('n/a')

    rep_sm_handles_list = []
    sm_post_volume_rank_string_list = []
    repost_rate_rank_string_list = []

    for row_tuple in mla_choices.iterrows():
        row = row_tuple[1]

        sm_handle, _, _, sm_post_volume_rank_string, _, repost_rate_rank_string, _, _, _, _, _, _ = get_member_sm_stuff(
            row, sm_posts_df, member_reposts, member_post_sentiment)

        # Remove the ranking bit
        sm_post_volume_rank_string = sm_post_volume_rank_string.split('<br />')[0]
        repost_rate_rank_string = repost_rate_rank_string.split('<br />')[0]
        
        rep_sm_handles_list.append(sm_handle)
        sm_post_volume_rank_string_list.append(sm_post_volume_rank_string)
        repost_rate_rank_string_list.append(repost_rate_rank_string)

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

    comb_rank_text = comb_rank_text.replace('1th ', '').replace('2th', '2nd').replace('3th', '3rd')

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
        #
        rep_names_list = normal_names_list,
        rep_parties_list = mla_choices.PartyName_long.tolist(),
        rep_party_colours_list = [party_colours[party_colours.party_name==p].colour.iloc[0] for p in mla_choices.PartyName],
        rep_roles_list = mla_choices.role.tolist(),
        rep_sm_handles_list = rep_sm_handles_list,
        rep_mla_or_mps_ids_list = mla_or_mp_ids_list,
        rep_image_urls_list = rep_image_urls_list,
        #
        sm_post_volume_rank_string_list = sm_post_volume_rank_string_list,
        repost_rate_rank_string_list = repost_rate_rank_string_list,
        #
        votes_present_string_list = votes_present_string_list,
        top_contrib_topic_list_list = top_contrib_topic_list_list,
        #
        full_mla_list = sorted_mlas_mps_list,
        postcodes_list = postcode_and_constits_list,
        #
        combined_demog_table_list = [e[1].values.tolist() for e in combined_demog_table[['constit','total_population',
            'mean_age','median_wage','pct_brought_up_protestant']].iterrows()],
        combined_demog_table2_list = [e[1].values.tolist() for e in combined_demog_table[['constit','total_area_sqkm',
            'pct_urban','n_farms','pct_work_in_agri','pct_adults_IS_claimants','pct_children_in_IS_households']].iterrows()],
        constit_population = f"{demogs_for_constit.total_population:,}",
        constit_second_message = comb_rank_text,
        constit_MDM_rank_order = combined_demog_table['MDM_mean_rank'].rank().astype(int).iloc[demogs_row_is_constit],
        constit_alphabetical_rank_order = combined_demog_table['constit'].rank().astype(int).iloc[demogs_row_is_constit],
        #
        elct_fcst_constit_party_stuff = elct_fcst_constit_party_stuff,
        cands_table_code = cands_table,
        elct_n_ensemble = elct_n_ensemble)


#
@app.route('/data/plot_minister_answer_times_<session_to_plot>')
def plot_minister_answer_times_fn_call(session_to_plot):
    if session_to_plot == CURRENT_ASSEMBLY_SESSION:
        plot_df = minister_time_to_answer
    else:
        plot_df = hist_minister_time_to_answer[hist_minister_time_to_answer.session_name==session_to_plot]
    return plot_minister_answer_times_fn(plot_df, party_colours)

#Most questions asked, split by written/oral
for session_to_plot in config['ALL_ASSEMBLY_SESSIONS']:
    if session_to_plot == CURRENT_ASSEMBLY_SESSION:
        questioners_df = questioners
    else:
        questioners_df = hist_questioners[hist_questioners.session_name==session_to_plot]
    create_route(f'/data/plot_questions_asked_{session_to_plot}_web', plot_questions_asked_fn, questioners_df, party_colours)
    create_route(f'/data/plot_questions_asked_{session_to_plot}_mobile', plot_questions_asked_fn, questioners_df, party_colours, mobile_mode=True)

#Votes PCA plot of all MLAs 
for session_to_plot in config['ALL_ASSEMBLY_SESSIONS']:
    if session_to_plot == CURRENT_ASSEMBLY_SESSION:
        plot_df = mlas_2d_rep
        pct_explained = (100*mla_votes_pca.explained_variance_ratio_[0], 100*mla_votes_pca.explained_variance_ratio_[1])
        current_session = True
    else:
        plot_df = hist_votes_pca_res[session_to_plot][0]
        pct_explained = hist_votes_pca_res[session_to_plot][1]
        current_session = False
    create_route(f'/data/plot_vote_pca_all_mlas_{session_to_plot}_web', plot_vote_pca_all_mlas_fn,
        plot_df, pct_explained, party_colours, current_session=current_session)
    create_route(f'/data/plot_vote_pca_all_mlas_{session_to_plot}_mobile', plot_vote_pca_all_mlas_fn,
        plot_df, pct_explained, party_colours, current_session=current_session, mobile_mode=True)

#Top plenary topics
@app.route('/data/plot_plenary_topics_overall_<session_to_plot>')
def plot_plenary_topics_overall_fn_call(session_to_plot):
    if session_to_plot == CURRENT_ASSEMBLY_SESSION:
        plot_df = plenary_contribs_topic_counts
    else:
        plot_df = hist_plenary_contribs_topic_counts[hist_plenary_contribs_topic_counts.session_name==session_to_plot]
    return plot_plenary_topics_overall_fn(plot_df, plenary_contribs_colour_dict)

#Plenary emotion scores
for session_to_plot in config['ALL_ASSEMBLY_SESSIONS']:
    if session_to_plot == CURRENT_ASSEMBLY_SESSION:
        emotions_party_df = emotions_party_agg
    else:
        emotions_party_df = hist_emotions_party_agg[hist_emotions_party_agg.session_name==session_to_plot]
    create_route(f'/data/plot_plenary_emotions_by_party_{session_to_plot}_web', plot_plenary_emotions_fn, emotions_party_df, party_colours)
    create_route(f'/data/plot_plenary_emotions_by_party_{session_to_plot}_mobile', plot_plenary_emotions_fn, emotions_party_df, party_colours, mobile_mode=True)


#Most sm posts by person
create_route('/data/plot_user_sm_post_num_web', plot_user_sm_post_num_fn, top_posters)
create_route('/data/plot_user_sm_post_num_mobile', plot_user_sm_post_num_fn, top_posters, mobile_mode=True)

#Highest sm repost rates
create_route('/data/plot_user_repost_web', plot_user_repost_fn, member_reposts, party_colours)
create_route('/data/plot_user_repost_mobile', plot_user_repost_fn, member_reposts, party_colours, mobile_mode=True)

#Highest and lowest sm post sentiment scores
create_route('/data/plot_user_sm_post_sentiment_web', plot_user_sm_post_sentiment_fn, member_post_sentiment, party_colours)
create_route('/data/plot_user_sm_post_sentiment_mobile', plot_user_sm_post_sentiment_fn, member_post_sentiment, party_colours, mobile_mode=True)


#Top news sources overall
create_route('/data/plot_news_sources_web', plot_news_sources_fn, news_sources)
create_route('/data/plot_news_sources_mobile', plot_news_sources_fn, news_sources, mobile_mode=True)

#News volume by party and week
create_route('/data/plot_news_volume_web', shared_plot_news_fn, news_sentiment_by_party_week, 'vol_smooth', 'Number of mentions', '', party_colours)
create_route('/data/plot_news_volume_mobile', shared_plot_news_fn, news_sentiment_by_party_week, 'vol_smooth', 'Number of mentions', '', party_colours, mobile_mode=True)


#Constituency metrics on the postcode page
@app.route('/data/plot_constituency_depriv_metrics_<constit_choice>_web')
def plot_constituency_depriv_metrics_fn_web(constit_choice):
    return plot_constituency_depriv_metrics_fn(combined_demog_table, constit_choice)
@app.route('/data/plot_constituency_depriv_metrics_<constit_choice>_mobile')
def plot_constituency_depriv_metrics_fn_mobile(constit_choice):
    return plot_constituency_depriv_metrics_fn(combined_demog_table, constit_choice, mobile_mode=True)

#Polls tracker
create_route('/data/polls_plot_web', polls_plot_fn, polls_df, elections_df, poll_avgs_track, party_colours)
create_route('/data/polls_plot_mobile', polls_plot_fn, polls_df, elections_df, poll_avgs_track, party_colours, mobile_mode=True)

if show_election_forecast:
    #Election cw pct bars
    create_route('/data/elct_cw_bars_plot_web', elct_cw_bars_fn, elct_fcst_cw_fps, party_colours)
    create_route('/data/elct_cw_bars_plot_mobile', elct_cw_bars_fn, elct_fcst_cw_fps, party_colours, mobile_mode=True)

    #Election cw seats bubble plot
    create_route('/data/elct_cw_seats_range_plot_web', elct_cw_seats_range_fn, elct_fcst_ens_party_seats, party_colours)
    create_route('/data/elct_cw_seats_range_plot_mobile', elct_cw_seats_range_fn, elct_fcst_ens_party_seats, party_colours, mobile_mode=True)

    #Election party delta pcts faceted by constit
    create_route('/data/elct_cw_delta_seats_map_plot_web', elct_cw_delta_seats_map_fn, elct_fcst_seat_deltas, party_colours)
    create_route('/data/elct_cw_delta_seats_map_plot_mobile', elct_cw_delta_seats_map_fn, elct_fcst_seat_deltas, party_colours, mobile_mode=True)

    #Election most seats ring plot
    create_route('/data/elct_cw_most_seats_plot_web', elct_cw_most_seats_fn, biggest_party_fracs)
    create_route('/data/elct_cw_most_seats_plot_mobile', elct_cw_most_seats_fn, biggest_party_fracs, mobile_mode=True)


@app.route('/robots.txt', methods=['GET'])
def static_from_root():
    return send_from_directory(app.static_folder, request.path[1:])


if __name__ == '__main__':
    if getpass.getuser() == 'david':
        app.run(debug=True)
    else:
        app.run(debug=False)  #don't need this for PythonAnywhere?


