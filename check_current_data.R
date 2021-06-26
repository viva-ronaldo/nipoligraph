#Check what records we now have in each source

suppressPackageStartupMessages(library(dplyr))
library(feather)
suppressPackageStartupMessages(library(lubridate))
library(jsonlite)
library(httr)

time_since_last_df <- data.frame(source=c(), days_since_last=c())

#People lists
politicians_list <- read.csv('data/all_politicians_list.csv')
cat(sprintf('\n%i politicans in all_politicians_list.csv; %i active (%i MLAs, %i MPs, %i other)\n',
            nrow(politicians_list), sum(politicians_list$active), sum(politicians_list$active & politicians_list$role=='MLA'),
            sum(politicians_list$active & politicians_list$role=='MP'), 
            sum(politicians_list$active & politicians_list$role=='other')))
twitter_list <- read.csv('data/politicians_twitter_accounts_ongoing.csv', colClasses = c('user_id'='character'))
cat(sprintf('%i Twitter accounts in politicians_twitter_accounts_ongoing.csv\n', nrow(twitter_list)))
cat(sprintf('Not tracking a Twitter account for: %s\n', paste(setdiff(subset(politicians_list, active==1)$normal_name, twitter_list$normal_name), collapse=', ')))

current_mlas_from_aims <- GET('http://data.niassembly.gov.uk/members.asmx/GetAllCurrentMembers_JSON?')
current_mlas_from_aims <- fromJSON(content(current_mlas_from_aims, as='text'))$AllMembersList$Member
if (length(setdiff(current_mlas_from_aims$PersonId, politicians_list$PersonId)) > 0) {
    cat(sprintf('\n**New MLA(s)! %s**\n\n', paste(subset(current_mlas_from_aims, !(PersonId %in% politicians_list$PersonId))$MemberName, collapse=', ')))
}

tmp <- read.csv('data/mla_email_addresses.csv')
missing_emails <- subset(politicians_list, role=='MLA' & active==1 & !(PersonId %in% tmp$PersonId))
if (nrow(missing_emails) > 0) {
    cat(sprintf('\n**Need to add MLA email addresses: %s**\n', paste(missing_emails$normal_name, collapse=', ')))
}
tmp <- names(read_json('data/mp_api_numbers.json'))
missing_mp_numbers <- subset(politicians_list, role=='MP' & active==1 & !(normal_name %in% tmp) & normal_name != 'Micky Brady') #took Brady out of list because he doesn't have a photo, so use other file for photo
if (nrow(missing_mp_numbers) > 0) {
    cat(sprintf('\n**Need to add MP ids: %s**\n', paste(missing_mp_numbers$normal_name, collapse=', ')))
}
tmp <- read.csv('data/party_colours.csv')
missing_party_colours <- subset(politicians_list, !(PartyName %in% tmp$party_name))
if (nrow(missing_party_colours) > 0) {
    cat(sprintf('\n**Need to add colour for a new party: %s (check party_group, short names files too)**\n\n', paste(missing_party_colours$PartyName, collapse=', ')))
}

#Assembly basics
tmp <- read.csv('data/diary_future_events.psv', sep='|')
cat(sprintf('\n%i events in Assembly diary; earliest is %s, latest is %s\n',
            nrow(tmp), ymd_hms(min(tmp$EventDate)), ymd_hms(max(tmp$EventDate))))
tmp <- read.csv('data/current_ministers_and_speakers.csv')
cat(sprintf('%i ministers + speakers\n', n_distinct(tmp$PersonId)))
tmp <- read.csv('data/current_committee_memberships.csv')
cat(sprintf('%i MLAs appear in Committees list, on %i committees\n', 
            n_distinct(tmp$PersonId), n_distinct(tmp$Organisation)))

#Assembly questions and answers
tmp <- read_feather('data/niassembly_questions_alltopresent.feather')
cat(sprintf('%i rows in questions file; %i unique questions; latest question tabled on %s\n', 
            nrow(tmp), n_distinct(tmp$DocumentId), strptime(ymd_hms(max(tmp$TabledDate)),'%Y-%m-%d')))
time_since_last_df <- rbind(time_since_last_df, data.frame(source='questions', days_since_last=as.integer(now()-ymd_hms(max(tmp$TabledDate)))))
any_new_mlas <- subset(tmp, !(TablerPersonId %in% politicians_list$PersonId))
if (nrow(any_new_mlas) > 0) {
    cat(sprintf('\n**%i questions from MLAs not in all_politicians list: %s**\n\n', nrow(any_new_mlas), paste(unique(any_new_mlas$TablerPersonId), collapse=', ')))
}

tmp <- read_feather('data/niassembly_answers_alltopresent.feather')
cat(sprintf('%i rows in answers file; %i unique answers; latest answer on %s\n', 
            nrow(tmp), n_distinct(tmp$DocumentId), strptime(ymd_hms(max(tmp$AnsweredOnDate)),'%Y-%m-%d')))
time_since_last_df <- rbind(time_since_last_df, data.frame(source='answers', days_since_last=as.integer(now()-ymd_hms(max(tmp$AnsweredOnDate)))))
any_new_mlas <- subset(tmp, !(MinisterPersonId %in% politicians_list$PersonId))
if (nrow(any_new_mlas) > 0) {
    cat(sprintf('\n**%i answers from Ministers not in all_politicians list: %s**\n\n', nrow(any_new_mlas), paste(unique(any_new_mlas$MinisterPersonId), collapse=', ')))
}


#Assembly debates
tmp <- read_feather('data/plenary_hansard_contribs_201920sessions_topresent.feather')
cat(sprintf('%i rows in debate contribs file, from %i sessions; latest date is %s\n',
            nrow(tmp), n_distinct(tmp$session_id), strptime(ymd_hms(max(tmp$PlenaryDate)),'%Y-%m-%d')))
time_since_last_df <- rbind(time_since_last_df, data.frame(source='debate contribs', days_since_last=as.integer(now()-ymd_hms(max(tmp$PlenaryDate)))))
any_new_mlas <- subset(tmp, !(speaker %in% politicians_list$normal_name))
if (nrow(any_new_mlas) > 0) {
    cat(sprintf('\n**%i debate contributions from MLAs not in all_politicians list: %s**\n\n', nrow(any_new_mlas), paste(unique(any_new_mlas$speaker), collapse=', ')))
}
tmp <- read_feather('data/plenary_hansard_contribs_emotions_averaged_201920sessions_topresent.feather')
cat(sprintf('%i MLAs in debate emotions file; %i excluding unknown-*\n', n_distinct(tmp$speaker),
            n_distinct(subset(tmp, !grepl('unknown',speaker))$speaker)))
tmp <- read.csv('data/lda_scored_plenary_contribs.csv')
cat(sprintf('%i rows in LDA scored debate contribs file, from %i sessions\n',
            nrow(tmp), n_distinct(tmp$session_id)))

#Votes
tmp <- read_feather('data/division_votes.feather')
cat(sprintf('%i division votes in current session; last one on %s', 
            n_distinct(tmp$EventId), strptime(max(tmp$DivisionDate),'%Y-%m-%d')))
time_since_last_df <- rbind(time_since_last_df, data.frame(source='votes', days_since_last=as.integer(now()-ymd_hms(max(tmp$DivisionDate)))))
tmp <- read_feather('data/division_vote_results.feather')
cat(sprintf('%i divisions with votes cast; %i votes total from %i MLAs\n',
            n_distinct(tmp$EventId), nrow(tmp), n_distinct(tmp$PersonId)))
any_new_mlas <- subset(tmp, !(PersonId %in% politicians_list$PersonId))
if (nrow(any_new_mlas) > 0) {
    cat(sprintf('\n**%i votes from MLAs not in all_politicians list: %s**\n\n', nrow(any_new_mlas), paste(unique(any_new_mlas$PersonId), collapse=', ')))
}

#News
tmp <- read_feather('data/newscatcher_articles_slim_w_sentiment_sep2020topresent.feather')
cat(sprintf('\n%i rows in (slim) news file covering %i people; latest date is %s; %i sources in last month\n',
            nrow(tmp), n_distinct(tmp$normal_name), strptime(max(tmp$published_date),'%Y-%m-%d'),
            n_distinct(subset(tmp, as.integer(difftime(Sys.Date(), published_date)) <= 30)$source)))
time_since_last_df <- rbind(time_since_last_df, data.frame(source='news', days_since_last=as.integer(now()-ymd_hms(max(tmp$published_date)))))

#Tweets
#mlas_2019_tweets_apr2019min_to_present.feather
tmp <- read_feather('data/mlas_2019_tweets_apr2019min_to_present_slim.feather')
tweets_last_month <- subset(tmp, as.integer(difftime(Sys.Date(), tmp$created_at)) <= 30)
cat(sprintf('\n%i tweets from %i people (in slim file); %i people with tweets in last month\n',
            nrow(tmp), n_distinct(tmp$user_id),
            n_distinct(tweets_last_month$user_id)))
time_since_last_df <- rbind(time_since_last_df, data.frame(source='tweets', days_since_last=as.integer(now()-ymd_hms(max(tmp$created_at)))))
inactive_twitter <- twitter_list$normal_name[(twitter_list$normal_name %in% subset(politicians_list, active==1)$normal_name) & 
    !(twitter_list$user_id %in% tweets_last_month$user_id)]
if (length(inactive_twitter) > 0) {
    cat(sprintf('\n%i Twitter accounts (for active politicians) with no tweets in last month: %s\n\n', length(inactive_twitter), paste(sort(inactive_twitter), collapse=', ')))
}
#tmp <- read.csv('data/tweets_network_last3months_top5s.csv')
tmp <- read.csv('data/vader_scored_tweets_apr2019min_to_present.csv')
cat(sprintf('%i tweets in Vader scored file\n', nrow(tmp)))
tmp <- read_json('flask/static/tweets_network_last3months_nodes.json')
cat(sprintf('%i people in the tweets graph\n\n', length(lapply(tmp, function(x) x$plot_label_all))))
tweeted_but_not_in_graph <- subset(twitter_list, user_id %in% tweets_last_month$user_id)$mla_name
tweeted_but_not_in_graph <- tweeted_but_not_in_graph[tweeted_but_not_in_graph %in% subset(politicians_list, active==1)$normal_name]
tweeted_but_not_in_graph <- tweeted_but_not_in_graph[!(tweeted_but_not_in_graph %in% as.character(lapply(tmp, function(x) x$plot_label)))]
if (length(tweeted_but_not_in_graph) > 0) {
    cat(sprintf('\n**%i people tweeted something in last month but are not in graph (may not have been retweeted by another politician): %s**\n\n',
                length(tweeted_but_not_in_graph), paste(tweeted_but_not_in_graph, collapse=', ')))
}

print(arrange(time_since_last_df, -days_since_last))

#polls
