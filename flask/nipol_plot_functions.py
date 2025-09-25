import datetime
import altair as alt
import pandas as pd

# --- Some plot functions ---

def add_grey_legend(plot, orient='top-right', columns=1, mobile_mode=False):
    """ """
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

# Social media

def plot_user_sm_post_num_fn(top_posters, mobile_mode=False):
    """ """
    top_posters_plot_df = (
        top_posters
        .merge(
            top_posters
            .groupby('normal_name', as_index=False)
            .agg(total_posts = ('n_posts', 'sum'))
            .sort_values('total_posts', ascending=False)
            .head(10 if mobile_mode else 15),
            on='normal_name', how='inner'
        )
    )
    # This probably wouldn't work for twitter without tweets_df
    # else:
    #     top_n_tweeters = (
    #         tweets_df[tweets_df.created_at_yweek.isin(last_5_yweeks_tweets)]
    #         .groupby('normal_name', as_index=False).status_id.count()
    #         .sort_values('status_id', ascending=False)
    #         .head(10 if mobile_mode else 15)
    #         .normal_name.tolist()
    #     )
    #     top_posters_plot_df = top_tweeters[top_tweeters.normal_name.isin(top_n_tweeters)]

    selection = alt.selection_single(on='mouseover', empty='all')
    plot = (
        alt.Chart(top_posters_plot_df).mark_bar()
        .add_selection(selection)
        .encode(
            y=alt.Y('n_posts', title='Number of posts'),
            x=alt.Y('normal_name', sort='-y', title=None),
            color = alt.Color(f'post_type',
                scale=alt.Scale(
                    domain=['original', 'repost'],
                    range=['Peru', 'SlateGrey']
                ), legend=alt.Legend(title="")),
            opacity = alt.condition(selection, alt.value(1), alt.value(0.4)),
            tooltip='tooltip_text:N')
        .properties(title = '', 
            width = 'container', 
            height = 200 if mobile_mode else 300, 
            background='none')
    )
    plot = add_grey_legend(plot, mobile_mode=mobile_mode)

    return plot.to_json()

def plot_user_repost_fn(member_reposts, party_colours, mobile_mode=False):
    """ """
    plot_df = member_reposts.head(10 if mobile_mode else 15)
    selection = alt.selection_single(on='mouseover', empty='all')
    plot = (
        alt.Chart(plot_df).mark_bar()
        .add_selection(selection)
        .encode(
            y=alt.Y('reposts_per_post', title='Reposts per post'),
            x=alt.Y('normal_name', sort='-y', title=None),
            color = alt.Color('mla_party', 
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(plot_df.mla_party)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(plot_df.mla_party)]['colour'].tolist()
                    ), legend=alt.Legend(title='')),
            tooltip='tooltip_text:N')
        .properties(title = '', 
            width = 'container', 
            height = 200 if mobile_mode else 300,
            background = 'none')
    )
    plot = add_grey_legend(plot, mobile_mode=mobile_mode)

    return plot.to_json()

def plot_user_sm_post_sentiment_fn(member_post_sentiment, party_colours, mobile_mode=False):
    """ """
    n_of_each_to_plot = 6 #if mobile_mode else 10
    df_to_plot = member_post_sentiment[member_post_sentiment.normal_name.isin(
        member_post_sentiment.sort_values('sentiment_vader_compound').head(n_of_each_to_plot).normal_name.tolist() +
        member_post_sentiment.sort_values('sentiment_vader_compound').tail(n_of_each_to_plot).normal_name.tolist()
    )].sort_values('sentiment_vader_compound')
    #df_to_plot['rank_group'] = ['highest']*n_of_each_to_plot + ['lowest']*n_of_each_to_plot

    #overall_av_sentiment_score = member_post_sentiment.sentiment_vader_compound.mean()
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

    base = (
        alt.Chart(df_to_plot)
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
    )
    #plot = base.mark_bar(size=3) + base.mark_circle(size=200, opacity=1)

    dividing_line = (
        alt.Chart(pd.DataFrame({'anything': [0]})).mark_rule(yOffset=0, strokeDash=[2,2])
       .encode(y=alt.Y('anything'))
    )
    #plot += dividing_line

    text = alt.Chart(df_to_plot).mark_text(
        align='center',
        baseline='middle',
        dx=0, 
        dy=0,
        fontSize = 13 if mobile_mode else 15
    ).encode(text='text', y='y2',
        x=alt.X('normal_name', sort=alt.EncodingSortField(field='sentiment_vader_compound', op='max')), 
    ).transform_filter(alt.datum.text != '')
    #plot += text

    plot = (
        (base.mark_bar(size=3) + base.mark_circle(size=200, opacity=1) + dividing_line + text)
        .properties(
            width = 'container', 
            height = 250 if mobile_mode else 400,
            background = 'none'
        )
    )
    plot = add_grey_legend(plot, orient='bottom-right', mobile_mode=mobile_mode)

    return plot.to_json()


# #Tweets PCA scatter of politicians}
# @app.route('/data/plot_tweet_pca_all_mlas_web')
# def plot_tweet_pca_all_mlas_fn_web():
#     return plot_tweet_pca_all_mlas_fn()

# @app.route('/data/plot_tweet_pca_all_mlas_mobile')
# def plot_tweet_pca_all_mlas_fn_mobile():
#     return plot_tweet_pca_all_mlas_fn(mobile_mode = True)

# def plot_tweet_pca_all_mlas_fn(mobile_mode=False):
#     """ """
#     plot = alt.Chart(tweet_pca_positions).mark_circle(opacity=0.6)\
#         .encode(x=alt.X('mean_PC1', 
#                     #axis=alt.Axis(title='Principal component 1 (explains 12% variance)', labels=False)),
#                     axis=alt.Axis(title=['<---- more asking for good governance','             more historical references or Irish language ---->'], labels=False)),
#             y=alt.Y('mean_PC2', #axis=alt.Axis(title='Principal component 2 (explains 2% variance)', labels=False)),
#                 axis=alt.Axis(title='more praising others <----    ----> more brexit and party politics', labels=False)),
#             color = alt.Color('mla_party', 
#                 scale=alt.Scale(
#                     domain=party_colours[party_colours.party_name.isin(tweet_pca_positions.mla_party)]['party_name'].tolist(), 
#                     range=party_colours[party_colours.party_name.isin(tweet_pca_positions.mla_party)]['colour'].tolist()
#                     ),
#                 legend=alt.Legend(title='')),
#             size = alt.Size('num_tweets', scale=alt.Scale(range=[30 if mobile_mode else 60, 250 if mobile_mode else 500]),
#                 legend=None),
#             tooltip = 'tooltip_text:N')\
#         .properties(title = '', 
#             width = 'container', 
#             height = 250 if mobile_mode else 450,  #stretch x-axis because PC1 explains more variance?
#             background = 'none')

#     if not mobile_mode:
#         plot = plot.encode(href = 'indiv_page_url:N')

#     plot = add_grey_legend(plot, orient = 'top', columns = 2 if mobile_mode else 4,
#         mobile_mode = mobile_mode)
#     #plot = plot.configure_axis(titleFontSize=10)

#     return plot.to_json()

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


# Polls

def polls_plot_fn(polls_df, elections_df, poll_avgs_track, party_colours, mobile_mode=False):
    """ """
    selection2 = alt.selection_single(fields=['pct_type'], bind='legend')

    cols_selection = ['date', 'pct', 'party', 'tooltip_text', 'pct_type']
    joint_plot_df = pd.concat([
        polls_df.assign(pct_type = 'polls')[cols_selection],
        elections_df.assign(pct_type = 'elections')[cols_selection]
    ])
    #Alternative to the scroll option is to only show last 3 years on mobile
    if False and mobile_mode:
        years_to_show_on_mobile = 3
        joint_plot_df = joint_plot_df[joint_plot_df.date >= (datetime.datetime.today() - datetime.timedelta(days=years_to_show_on_mobile*365))]
        poll_avgs_track_plot_df = poll_avgs_track[poll_avgs_track.date >= (datetime.datetime.today() - datetime.timedelta(days=years_to_show_on_mobile*365))]
    else:
        poll_avgs_track_plot_df = poll_avgs_track

    plot1 = (
        alt.Chart(joint_plot_df)
        .mark_point(filled=True, size=100 if mobile_mode else 150)
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
            tooltip = 'tooltip_text:N')
        .interactive(bind_x=True, bind_y=False)
        .add_selection(selection2)
    )

    #Draw lines with separate data to get control on size
    plot2 = (
        alt.Chart(poll_avgs_track_plot_df)
        .mark_line(size=3)
        .encode(x='date:T', y='pred_pct',
            color = alt.Color('party',
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(poll_avgs_track_plot_df.party)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(poll_avgs_track_plot_df.party)]['colour'].tolist()
                    ), 
                legend = alt.Legend(title = ''))
            )
    )

    plot_combined = (
        alt.layer(plot2, plot1)
        .configure_axis(titleFontSize=14, labelFontSize=10)
        .configure_legend(
            direction='horizontal', 
            orient='top-right',
            strokeColor='gray',
            fillColor='#EEEEEE',
            padding=10,
            cornerRadius=10,
            columns=4,
            #columns = 2 if mobile_mode else 4,  #if not using scroll, use this
            labelFontSize=8 if mobile_mode else 12,
            symbolSize=40 if mobile_mode else 60
        )
        .properties(title='', 
            width=600 if mobile_mode else 'container',  #rely on horizontal scroll on mobile
            height=400,
            background='none')
    )

    return plot_combined.to_json()

# Assembly

def plot_questions_asked_fn(questioners_df, party_colours, mobile_mode=False):
    """ """
    plot_df = (
        questioners_df
        .sort_values('Questions asked', ascending=False)
        .groupby('Question type')
        .head(8 if mobile_mode else 12)
    )

    plot = (
        alt.Chart(plot_df)
        .mark_bar(opacity=1)
        .encode(
            y=alt.Y('Questions asked'),
            x=alt.X('normal_name', sort='-y', title=None),
            color = alt.Color('PartyName',
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(plot_df.PartyName)]['party_name'].tolist(),
                    range=party_colours[party_colours.party_name.isin(plot_df.PartyName)]['colour'].tolist()
                    ), legend=None),
            facet = alt.Facet('Question type:N', columns=1),
            tooltip='tooltip_text:N')
        .resolve_scale(x='independent', y='independent')
        .properties(title = '',
            width=220 if mobile_mode else 440, 
            height=250,
            background='none')
    )

    return plot.to_json()

def plot_minister_answer_times_fn(minister_time_to_answer_df, party_colours):
    """ """
    #Problem with sorting two layer chart in Vega
    #Fixed in Altair 3 according to here https://github.com/altair-viz/altair/issues/820
    #  but seems not to be in my case
    #Sorting values first also seems not to work
    #Another workaround is to do second layer with independent axis, and hide labels and ticks,
    #  and this does work (https://github.com/altair-viz/altair/issues/820#issuecomment-386856394)
    filtered_party_colours = party_colours[party_colours.party_name.isin(minister_time_to_answer_df.Minister_party_name)]
    plot_a = (
        alt.Chart(minister_time_to_answer_df)
        .mark_bar(size=3)
        .encode(x=alt.X('Questions answered', axis=alt.Axis(grid=True)),
            y=alt.Y('Minister_normal_name', sort=alt.EncodingSortField(order='ascending', field='Questions answered'),
                axis = alt.Axis(title=None)),
            color = alt.Color('Minister_party_name', 
                scale=alt.Scale(
                    domain=filtered_party_colours['party_name'].tolist(), 
                    range=filtered_party_colours['colour'].tolist()
                    )))
        #.properties(title = ' ', width='container', height=250)
    )
    
    # #Lose the axis ordering if add this on top
    #plot_b = plot.mark_circle(size=80)#\
        #.encode(x='Median days to answer',
        #    y=alt.Y('Minister_normal_name', sort=alt.EncodingSortField(order='ascending', field='tmp_sort_field')))
            #y=alt.Y('Minister_normal_name', sort='x'),
    #plot = plot_a + plot_b

    #default opacity is < 1 for circles so have to set to 1 to match bars
    plot_b = (
        alt.Chart(minister_time_to_answer_df)
        .mark_circle(size=200, opacity=1)
        .encode(x='Questions answered',
            y=alt.Y('Minister_normal_name', sort=alt.SortField(order='ascending', field='Questions answered'),
                axis = alt.Axis(labels=False, ticks=False, title=None)),
            color = alt.Color('Minister_party_name', 
                scale=alt.Scale(
                    domain=filtered_party_colours['party_name'].tolist(), 
                    range=filtered_party_colours['colour'].tolist()
                    )),
            size = 'Median days to answer',
            tooltip = 'tooltip_text')
        #.properties(title = '', width=300, height=250)
    )

    plot = (
        alt.layer(plot_a, plot_b, data=minister_time_to_answer_df)
        .resolve_scale(y='independent')
        .properties(width='container', height=300, background='none')
        .configure_legend(disable=True)
    )

    return plot.to_json()


def plot_vote_pca_all_mlas_fn(plot_df, pct_explained, party_colours, current_session=True, mobile_mode=False):
    """ """
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

    if current_session:
        plot = plot.encode(href='indiv_page_url:N')

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

def plot_plenary_topics_overall_fn(plot_df, plenary_contribs_colour_dict):
    """ """
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

def plot_plenary_emotions_fn(emotions_party_df, party_colours, mobile_mode=False):
    """ """
    selection_by_party = alt.selection_single(on='mouseover', empty='all', encodings=['color'])

    #TODO would be nice to have on hover, selected party vs average of the rest

    emotions_list = ['trust', 'anticipation', 'joy', 'sadness', 'anger']
    parties_to_plot = ['Alliance', 'DUP', 'SDLP', 'Sinn Fein', 'UUP']
    #fear and disgust are strongly correlated with anger and sadness, so can omit
    #surprise values are very similar for the 5 parties
    emotions_party_to_plot = (
        emotions_party_df[
            (emotions_party_df.emotion_type.isin(emotions_list)) &
            (emotions_party_df.PartyName.isin(parties_to_plot))
        ].merge(
            pd.DataFrame({'order': [1, 2, 3, 4, 5], 'emotion_type': emotions_list}),
            on='emotion_type', how='inner'
        )
    )

    plot = (
        alt.Chart(emotions_party_to_plot)
        .mark_point(size=120 if mobile_mode else 200, strokeWidth=5 if mobile_mode else 6)
        .add_selection(selection_by_party)
        .encode(
            x = alt.X('ave_emotion', title='Fraction words scoring'),
            y = alt.Y('emotion_type', title=None,
                sort=alt.EncodingSortField(order='ascending', field='order')),
            color = alt.Color('PartyName', 
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(emotions_party_to_plot.PartyName)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(emotions_party_to_plot.PartyName)]['colour'].tolist()
                ), legend=None),
            opacity = alt.condition(selection_by_party, alt.value(0.7), alt.value(0.15)),
            tooltip = 'PartyName'
        )
        .properties(height = 200 if mobile_mode else 250, 
            width='container',
            background='none')
    )

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


# News

def plot_news_sources_fn(news_sources, mobile_mode=False):
    """ """
    topN_sources = (
        news_sources
        .groupby('source', as_index=False).agg({'News articles': sum})
        .sort_values('News articles').tail(10 if mobile_mode else 15)
        .source.tolist()
    )
    news_sources_plot_df = news_sources[news_sources.source.isin(topN_sources)]

    selection = alt.selection_single(on='mouseover', empty='all')
    plot = (
        alt.Chart(news_sources_plot_df).mark_bar()
        .add_selection(selection)
        .encode(
            y=alt.Y('News articles'),
            x=alt.X('source', sort='-y', title=None),
            color = alt.Color('PartyGroup', 
                scale=alt.Scale(
                    domain=['Unionist','Other','Nationalist'],
                    range=['RoyalBlue','Moccasin','LimeGreen']
                ), legend = alt.Legend(title = 'Mentioning')),
            opacity = alt.condition(selection, alt.value(1), alt.value(0.4)),
            tooltip='tooltip_text:N')
        .configure_axis(labelAngle=-45)
        .properties(title = '', 
            width = 'container', 
            height = 200 if mobile_mode else 300,
            background = 'none')
    )
    plot = add_grey_legend(plot, mobile_mode=mobile_mode)

    return plot.to_json() 

def shared_plot_news_fn(news_sentiment_by_party_week, y_variable, y_title, title, party_colours, mobile_mode=False):
    """ Plot time series of news volume or sentiment from news_sentiment_by_party_week """
    news_sentiment_by_party_week['plot_date'] = news_sentiment_by_party_week.apply(
        lambda row: pd.to_datetime('{}-01-01'.format(row['published_date_year'])) + pd.to_timedelta(7*(row['published_date_week']-1), unit='days'),
        axis=1)

    plot = (
        alt.Chart(news_sentiment_by_party_week).mark_line(size=5)
        .encode(x=alt.X('plot_date:T', title=None),
            y=alt.Y(y_variable, axis=alt.Axis(title=y_title, minExtent=10, maxExtent=100)),
            color = alt.Color('PartyName', 
                scale=alt.Scale(
                    domain=party_colours[party_colours.party_name.isin(news_sentiment_by_party_week.PartyName)]['party_name'].tolist(), 
                    range=party_colours[party_colours.party_name.isin(news_sentiment_by_party_week.PartyName)]['colour'].tolist()
                    ),
                legend=alt.Legend(title='')))
        .configure_axis(labelAngle=-30)
        .properties(title=title,
            width='container', 
            height=200 if mobile_mode else 300,
            background='none')
        .configure_legend(
            direction='horizontal', 
            orient='top',
            strokeColor='gray',
            fillColor='#EEEEEE',
            padding = 7 if mobile_mode else 10,
            cornerRadius = 10,
            columns = 2 if mobile_mode else 5
        )
    )

    if not mobile_mode:
        plot = plot.configure_axis(labelFontSize=14)

    return plot.to_json()

# @app.route('/data/plot_news_sentiment')
# def plot_news_sentiment_fn(mobile_mode = False):
#     #There seems to be a bug in Vega boxplot continuous title, possibly 
#     #  related to it being a layered plot - gives title of 'variable, mytitle, mytitle'
#     #title=None gives no title; title='' gives title=variable name
#     #news_sentiment_by_party_week['Sentiment score'] = news_sentiment_by_party_week.sr_sentiment_score
#     #work out order manually - easier than figuring out the altair method
#     party_order = news_sentiment_by_party_week.groupby('PartyName').sr_sentiment_score.mean().sort_values(ascending=False).index.tolist()

#     plot = alt.Chart(news_sentiment_by_party_week).mark_boxplot(size=30)\
#         .encode(y = alt.Y('PartyName:N', title=None, sort=party_order),
#             #x = alt.X('Sentiment score:Q', title=''),
#             x = alt.X('sr_sentiment_score', title=None),
#             color = alt.Color('PartyName', 
#                 scale=alt.Scale(
#                     domain=party_colours[party_colours.party_name.isin(news_sentiment_by_party_week.PartyName)]['party_name'].tolist(), 
#                     range=party_colours[party_colours.party_name.isin(news_sentiment_by_party_week.PartyName)]['colour'].tolist()
#                     ),
#                 legend=None),
#             tooltip=alt.Tooltip(field='tooltip_text', type='nominal', aggregate='max'))\
#         .properties(title = '',
#             width='container', 
#             height = 200 if mobile_mode else 300,
#             background = 'none')
#     #use the aggregated tooltip to avoid printing the full summary
#     #  with long floats and variable names

#     plot = plot.configure_legend(
#         direction='horizontal', 
#         orient='top',
#         strokeColor='gray',
#         fillColor='#EEEEEE',
#         padding=10,
#         cornerRadius=10
#     )

#     return plot.to_json()


# Postcodes

def plot_constituency_depriv_metrics_fn(combined_demog_table, constit_choice, mobile_mode=False):
    """ """
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
    
    plot = (
        (plot + annotations)
        .properties(
            #title=f'Deprivation scores for {constit_choice} vs other constituencies',
            title='',
            height = 250 if mobile_mode else 350, 
            width = 'container',
            background = 'none'
        )
        .configure_axis(titleFontSize=14, labelFontSize=13, grid=False)
        .configure_title(fontSize=18, font='Arial')
    )

    return plot.to_json()


# Election forecast ones probably won't be used again

def elct_cw_bars_fn(elct_fcst_cw_fps, party_colours, mobile_mode=False):
    """ """
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

def elct_cw_seats_range_fn(elct_fcst_ens_party_seats, party_colours, mobile_mode=False):
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

def elct_cw_delta_seats_map_fn(elct_fcst_seat_deltas, party_colours, mobile_mode=False):

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

def elct_cw_most_seats_fn(biggest_party_fracs, mobile_mode=False):
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

