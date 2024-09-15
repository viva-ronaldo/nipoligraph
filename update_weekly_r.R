# Weekly update steps
# i) Check for new politicians from AIMS
# ii) Assembly diary, committees list, ministers list
# iii) Questions and answers
# iv) Division votes ('votes' (vote details) and 'vote_results' (how members voted))
# v) Contributions by each member in Hansard
# vi) Average emotions for current session contributions
# SKIPPED NOW: Get extra news articles from Twitter (not in NewsCatcher)
# viii) Sentiment for news article mentions - TODO replace/enhance with GPT summary
# SKIPPED NOW: Get tweets, calculate tweet network

suppressPackageStartupMessages(library(dplyr))
library(httr)
library(jsonlite)
#library(feather)
suppressPackageStartupMessages(library(arrow))  # for read_feather
suppressPackageStartupMessages(library(rtweet))
library(stringr)
library(tidytext)
library(magrittr)
library(sentimentr)
suppressPackageStartupMessages(library(network))
suppressPackageStartupMessages(library(GGally))
suppressPackageStartupMessages(library(igraph))
library(intergraph)

if (Sys.info()['user']=='rstudio') setwd('/home/rstudio/nipol')

source('./functions_for_update_weekly.R')

data_dir <- './data/'

current_session_name <- '2022-2027'  # e.g. '2022-2027' (made up by me)
current_min_session_name_for_plenary <- '2022-2023'  # e.g. 2022-2023 for the larger 2022-2027 session
do_twitter <- FALSE   # as of mid-2023, there is no affordable way of scraping latest tweets

# People ----
#i) First check for any new MLAs and add to the politicians file, then return the list
politicians_list_filepath <- file.path(data_dir, 'all_politicians_list.csv')
politicians <- update_and_get_politicians_list(politicians_list_filepath)

#ii) Latest diary events - take anything from current date to a year ahead
message('\nDoing Assembly diary, committees, minister lists, register of interests')
diary_filepath <- file.path(data_dir, 'diary_future_events.psv')
committees_list_filepath <- file.path(data_dir, 'current_committee_memberships.csv')
minister_list_filepath <- file.path(data_dir, 'current_ministers_and_speakers.csv')
mla_interests_list_filepath <- file.path(data_dir, 'current_mla_registered_interests.csv')
update_assembly_lists(
    diary_filepath,
    committees_list_filepath,
    minister_list_filepath,
    mla_interests_list_filepath
)
# TODO check for change in minister AffiliationTitle strings?


# Questions and answers ----

questions_filepath <- file.path(data_dir, 'niassembly_questions.feather')
answers_filepath <- file.path(data_dir, 'niassembly_answers.feather')
update_questions_and_answers(questions_filepath, answers_filepath)

# Votes ----

message('\nDoing votes')
vote_details_filepath <- file.path(data_dir, 'division_votes.feather')
vote_results_filepath <- file.path(data_dir, 'division_vote_results.feather')
update_vote_list(vote_details_filepath, vote_results_filepath)

# Contributions ----

message('\nDoing plenary contributions')
# plenary_hansard_contribs.feather started in 2022-05 - we want to use the current Assembly session
#   so this is OK; TODO get method for archiving older contribs, or use one file and filter in time in update_average_contrib_emotions
contribs_filepath <- file.path(data_dir, 'plenary_hansard_contribs.feather')
contribs_emotions_filepath <- file.path(data_dir, 'plenary_hansard_contribs_emotions_averaged.feather')

update_plenary_contribs(contribs_filepath,
                        current_session_name,
                        politicians_list_filepath,
                        file.path(data_dir, 'hist_mla_ids_by_session.feather'))

update_average_contrib_emotions(contribs_filepath, contribs_emotions_filepath)


# News article mentions ----

# The main news scraping is done in update_newscatcher_in_chunks_r.R
#   Irish News was done via twitter, but in 2023 this must be skipped
# Sentiment slim file is computed for all articles (via NewsCatcher and anything else);
#   TODO update with a GPT solution

news_articles_filepath <- file.path(data_dir, 'newscatcher_articles_sep2020topresent.feather')

if (do_twitter) {
    update_news_from_twitter(politicians, news_articles_filepath)
}

#Add sentiment on the summaries (newscatcher_articles file) which is updated elsewhere before this script runs
message('\nDoing news article sentiment')
update_news_article_sentiment(news_articles_filepath,
                              file.path(data_dir, 'newscatcher_articles_slim_w_sentiment_sep2020topresent.feather'))


# Tweets ----

if (do_twitter) {
    twitter_accounts_list_filepath <- file.path(data_dir, 'politicians_twitter_accounts_ongoing.csv')
    hist_tweets_filepath <- file.path(data_dir, 'tweets_slim_apr2019min_to_3jun2021.feather')
    current_tweets_filepath <- file.path(data_dir, 'tweets_full_4jun2021_to_present.feather')
    current_tweets_slim_filepath <- file.path(data_dir, 'tweets_slim_4jun2021_to_present.feather')
    
    twitter_ids <- read.csv(twitter_accounts_list_filepath,
                            stringsAsFactors = FALSE,
                            colClasses = list('user_id'='character', 'account_created_at'='Date')) %>%
        semi_join(politicians, by='normal_name')
    #Track tweets for anyone in the politicians list, active or inactive
    #Keep trying some accounts that are private but could change - currently Stalford, C Kelly
    
    #Removed entries from the list:
    #5/4/21: John Dallat,SDLP,829517565084983296,John Dallat MLA,johndallatmla,2017-02-09 02:29:34,2,1 (no tweets found anyway)
    
    #Skipping these in search below, but keep in the file because they have tweets in past and need their ids
    #6/6/21: Paula Bradley,DUP,1079572046,Paula Bradley,PaulaBradleyMLA,2013-01-11 12:48:46,12,2467
    #6/6/21: John Stewart,UUP,262862076,John Stewart,JohnStewart1983,2011-03-08 22:33:38,22,14638
    
    #Added to the list:
    #6/6/21: John Stewart,UUP,1391853244016644100,John Stewart,johnstewartuup,2021-05-10 20:30:57,20,219
    
    #Other notes:
    #- Late May 2021 user_id 275799277 changed screen_name from DUPleader to ArleneFosterUK; no problem here
    #- paulabradleymla deleted account ~March 2021
    #- May 2021 John Stewart closed old account and started as johnstewartuup, user_id 1391853244016644100
    #- Missed some in early August to late October 2022 because the update didn't run and won't be able to
    #    get them all retrospectively for the big users.
    
    twitter_ids <- subset(twitter_ids, !(user_id %in% c('262862076', #Paula Bradley, deleted
                                                        '1079572046', #John Stewart, deleted and replaced
                                                        '1488006216')  #Christopher Stalford, deceased, was private anyway
                                         ))
    cat(sprintf('Get tweets for %i users\n', nrow(twitter_ids)))
    
    old_twitter_method <- function(twitter_ids) {
        # ## create token named "twitter_token"
        # twitter_token <- create_token(
        #     app = appname,
        #     consumer_key = consumer_key,
        #     consumer_secret = consumer_secret,
        #     access_token = access_token,
        #     access_secret = access_secret)
        if (Sys.info()['user']=='rstudio') Sys.setenv('TWITTER_PAT'='/home/rstudio/nipol/.rtweet_token11.rds')
        #That saves something at /home/david/.rtweet_token11.rds, the path to which is in $TWITTER_PAT,
        #  so that next time I can do just:
        get_token()
        
        previous_mla_tweets <- read_feather(hist_tweets_filepath)[, c('status_id', 'created_at')]
        existing_mla_tweets <- read_feather(current_tweets_filepath)
        cat(sprintf('Have %s tweets currently\n',
                    prettyNum(nrow(existing_mla_tweets) + nrow(previous_mla_tweets), big.mark=',')))
        
        go_back_to_time <- max(existing_mla_tweets$created_at)
        #new_mla_tweets <- get_all_tweets_back_to_date(mla_ids, go_back_to_time)
        #can maybe just do in one go if guaranteeing noone will have more than 1000 (in a week)
        #  About 600 should be enough to cover Aiken, Long
        cat('Get Twitter users 1-20\n')
        new_mla_tweets <- tweets_data(get_timelines(user = twitter_ids$user_id[1:20], n = 600))  #get all at once
        cat('Get Twitter users 21-40\n')
        new_mla_tweets <- rbind(new_mla_tweets, tweets_data(get_timelines(user = twitter_ids$user_id[21:40], n = 600)))
        cat('Get Twitter users 41-60\n')
        new_mla_tweets <- rbind(new_mla_tweets, tweets_data(get_timelines(user = twitter_ids$user_id[41:60], n = 600)))
        cat('Get Twitter users 61-80\n')
        new_mla_tweets <- rbind(new_mla_tweets, tweets_data(get_timelines(user = twitter_ids$user_id[61:80], n = 600)))
        cat('Get Twitter users 81-end\n')
        new_mla_tweets <- rbind(new_mla_tweets, tweets_data(get_timelines(user = twitter_ids$user_id[81:length(twitter_ids$user_id)], n = 600)))
        new_mla_tweets <- new_mla_tweets[new_mla_tweets$created_at >= go_back_to_time, ]
        new_mla_tweets <- new_mla_tweets[!duplicated(new_mla_tweets), ]
        new_mla_tweets <- new_mla_tweets[, c('user_id','status_id','created_at','screen_name',
                                     'text','source','display_text_width','reply_to_status_id',
                                     'reply_to_user_id','is_quote','is_retweet','favorite_count',
                                     'retweet_count','hashtags','mentions_user_id','mentions_screen_name',
                                     'quoted_status_id','retweet_created_at','retweet_user_id',
                                     'retweet_screen_name','retweet_description')]
        new_mla_tweets$text <- plain_tweets(new_mla_tweets$text)
        return(new_mla_tweets)
    }
    
    #combine old screen_names from ids file with any new ones picked up this week
    any_mla_screen_names <- unique(c(twitter_ids$screen_name, new_mla_tweets$screen_name))
    new_mla_tweets$mentions_another_mla <- sapply(new_mla_tweets$mentions_screen_name, 
                                              function(m) as.logical(length(intersect(unlist(strsplit(m,', '))[[1]], any_mla_screen_names) > 0)))
    #unlist hashtags mentions_user_id, mentions_screen_name to save to feather
    new_mla_tweets$hashtags <- sapply(new_mla_tweets$hashtags, function(x) paste(x, collapse='|'))
    new_mla_tweets$mentions_user_id <- sapply(new_mla_tweets$mentions_user_id, function(x) paste(x, collapse='|'))
    new_mla_tweets$mentions_screen_name <- sapply(new_mla_tweets$mentions_screen_name, function(x) paste(x, collapse='|'))
    new_mla_tweets$created_ym <- sub('-','',substr(new_mla_tweets$created_at,1,7))
    
    #new_mla_tweets <- add_pca_scores_to_tweets(new_mla_tweets) 
    #Now doing in python using word vectors
    
    #all_mla_tweets <- rbind(existing_mla_tweets, new_mla_tweets)
    #all_mla_tweets <- all_mla_tweets[!duplicated(all_mla_tweets$status_id), ]
    new_mla_tweets <- subset(new_mla_tweets, !(status_id %in% previous_mla_tweets$status_id))
    new_mla_tweets <- subset(new_mla_tweets, !(status_id %in% existing_mla_tweets$status_id))
    all_mla_tweets <- rbind(existing_mla_tweets, new_mla_tweets)
    write_feather(all_mla_tweets, current_tweets_filepath)
    #and slimmer file for upload
    message('Writing slim tweets file')
    write_feather(all_mla_tweets[, c('user_id', 'screen_name', 'status_id', 'created_at', 'created_ym',
                                     'text', 'is_retweet', 'quoted_status_id', 'retweet_count')], 
                  current_tweets_slim_filepath)

    #Tweet network plot
    
    network_nodes_filepath <- 'flask/static/tweets_network_last3months_nodes.json'
    network_edges_filepath <- 'flask/static/tweets_network_last3months_edges.csv'
    network_top5s_filepath <- file.path(data_dir, 'tweets_network_last3months_top5s.csv')
    
    # TODO return to when have Twitter scraping working again
    #make_twitter_network_files(twitter_ids, network_nodes_filepath, network_edges_filepath, network_top5s_filepath)
    
}