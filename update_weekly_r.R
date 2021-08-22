library(dplyr)
library(httr)
library(jsonlite)
library(feather)
library(rtweet)
library(stringr)
library(tidytext)
library(magrittr)
library(sentimentr)
library(network)
library(GGally)

if (Sys.info()['user']=='rstudio') setwd('/home/rstudio/nipol')

source('./functions_for_update_weekly.R')

#Regular update: ----
#DONE diary
#DONE minister list
#DONE committee memberships
#DONE questions by each member
#DONE answers by each member
#DONE votes and how each member voted
#DONE contributions by each member in Hansard
#DONE news articles
#DONE tweets
#DONE score new tweets with PCA model
#DONE score tweets sentiment in python
#DONE tweet network

politicians <- read.csv('data/all_politicians_list.csv')

#Latest diary events - take anything from current date to 2025 ----
update_assembly_lists()

#Questions and answers ----

#Get questions with different method than originally, to use time filtering; go back further than should be necessary, just in case
tmp <- GET(sprintf('http://data.niassembly.gov.uk/questions.asmx/GetQuestionsForOralAnswer_TabledInRange_JSON?startDate=%s&endDate=%s',
                   Sys.Date()-21, '2025-01-01'))
tmp <- fromJSON(content(tmp, as='text'))$QuestionsList$Question

tmp2 <- GET(sprintf('http://data.niassembly.gov.uk/questions.asmx/GetQuestionsForWrittenAnswer_TabledInRange_JSON?startDate=%s&endDate=%s',
                   Sys.Date()-21, '2025-01-01'))
tmp2 <- fromJSON(content(tmp2, as='text'))$QuestionsList$Question

new_questions <- data.frame()
if (!is.null(tmp)) {
    tmp$RequestedAnswerType <- 'oral'
    new_questions <- rbind(new_questions,
                           tmp %>% select(DocumentId, TabledDate, TablerPersonId, MinisterPersonId,
                                          QuestionText, RequestedAnswerType))
}
if (!is.null(tmp2)) {
    tmp2$RequestedAnswerType <- 'written'
    new_questions <- rbind(new_questions,
                           tmp2 %>% select(DocumentId, TabledDate, TablerPersonId, MinisterPersonId,
                                           QuestionText, RequestedAnswerType))
}
questions_file_path <- 'data/niassembly_questions_alltopresent.feather'
existing_questions <- read_feather(questions_file_path)
if (nrow(new_questions) > 0) {
    existing_questions <- rbind(existing_questions, new_questions) %>% filter(!duplicated(.[,c('DocumentId','MinisterPersonId')]))
    message('Writing Assembly questions')
    write_feather(existing_questions, questions_file_path)
    rm(new_questions, tmp, tmp2)
} 

#Get answers with function
answers_file_path <- 'data/niassembly_answers_alltopresent.feather'
existing_answers <- read_feather(answers_file_path)
pending_questions <- existing_questions$DocumentId[!(existing_questions$DocumentId %in% existing_answers$DocumentId)]
if (length(pending_questions) > 0) {
    new_answers <- get_answers_to_questions(pending_questions)
}
existing_answers <- rbind(existing_answers, new_answers)
message('Writing Assembly answers')
write_feather(existing_answers, answers_file_path)

#Votes ----
message('Doing votes')
update_vote_list('data/division_votes.feather', 'data/division_vote_results.feather')

#Contributions ----
contribs_filepath <- 'data/plenary_hansard_contribs_201920sessions_topresent.feather'
existing_contribs <- read_feather(contribs_filepath)
tmp <- GET('http://data.niassembly.gov.uk/hansard.asmx/GetAllHansardReports_JSON?')
new_reports_list <- fromJSON(content(tmp, as='text'))$AllHansardComponentsList$HansardComponent %>%
    filter(PlenarySessionName >= '2019-2020', !(ReportDocId %in% existing_contribs$ReportDocId))
new_contribs <- data.frame()
for (doc_id in new_reports_list$ReportDocId) {
    session_hansard_contribs <- get_tidy_hansardcomponents_object(doc_id, '2020-2022')
    #at least one plenary was just two minutes' silence so returns 0 rows
    if (nrow(session_hansard_contribs) > 0) {
        session_hansard_contribs$ReportDocId <- doc_id
        session_hansard_contribs <- left_join(session_hansard_contribs, new_reports_list, by='ReportDocId')
    }
    new_contribs <- rbind(new_contribs, session_hansard_contribs)
}
#tail(sort(table(new_contribs$speaker, useNA='ifany')), 10)
message('Writing assembly contribs')
write_feather(rbind(existing_contribs, new_contribs), contribs_filepath)

#Update the averaged emotions file
#May need to upgrade the memory; for now, do in chunks
contribs <- rbind(existing_contribs, new_contribs)
speakers <- unique(contribs$speaker)
mla_emotions <- contribs %>% filter(speaker %in% speakers[1:30]) %>% 
        mutate(split_text = get_sentences(contrib)) %$%
        emotion_by(split_text, by=speaker)
message('Done plenary emotion chunk 1')
mla_emotions <- rbind(mla_emotions,
    contribs %>% filter(speaker %in% speakers[31:60]) %>% 
        mutate(split_text = get_sentences(contrib)) %$%
        emotion_by(split_text, by=speaker))
message('Done plenary emotion chunk 2')
mla_emotions <- rbind(mla_emotions,
    contribs %>% filter(speaker %in% speakers[61:length(speakers)]) %>% 
        mutate(split_text = get_sentences(contrib)) %$%
        emotion_by(split_text, by=speaker))
message('Done plenary emotion chunk 3')
message('Writing plenary contribs with emotions')
write_feather(mla_emotions, 'data/plenary_hansard_contribs_emotions_averaged_201920sessions_topresent.feather')

rm(existing_contribs, new_contribs)

#News article mentions ----
#Articles file is now updated separately before this script runs

#Add sentiment on the summaries, now updated elsewhere
news_articles <- read_feather('data/newscatcher_articles_sep2020topresent.feather')

news_articles$article_id <- seq_along(news_articles$normal_name)
news_sentiments <- news_articles %>%
    mutate(summary_split = get_sentences(summary)) %$%
    sentiment_by(summary_split, list(normal_name,article_id),
                 polarity_dt = lexicon::hash_sentiment_jockers_rinker %>% filter(!grepl('^econo|justice|money|assembly|traditional|socialist|progressive|conservative|voter|elect|holiday|guardian|star|independence|united',x)))
news_articles <- left_join(news_articles,
                           news_sentiments %>% select(article_id, sr_sentiment_score=ave_sentiment),
                           by='article_id') %>%
    select(-article_id)
message('Writing news slim sentiment')
write_feather(news_articles[,c('normal_name','published_date','source','link','title','sr_sentiment_score')],
              'data/newscatcher_articles_slim_w_sentiment_sep2020topresent.feather')


#Tweets ----

twitter_ids <- read.csv('data/politicians_twitter_accounts_ongoing.csv', stringsAsFactors = FALSE,
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
#Late May 2021 user_id 275799277 changed screen_name from DUPleader to ArleneFosterUK; no problem here
#paulabradleymla deleted account ~March 2021
#May 2021 John Stewart closed old account and started as johnstewartuup, user_id 1391853244016644100

twitter_ids <- subset(twitter_ids, !(user_id %in% c('262862076', #Paula Bradley, deleted
                                                    '1079572046') #John Stewart, deleted and replaced
                                     ))

# ## create token named "twitter_token"
# twitter_token <- create_token(
#     app = appname,
#     consumer_key = consumer_key,
#     consumer_secret = consumer_secret,
#     access_token = access_token,
#     access_secret = access_secret)
if (Sys.info()['user']=='rstudio') Sys.setenv('TWITTER_PAT'='/home/rstudio/nipol/.rtweet_token11.rds')
#That saves something at /home/david/.rtweet_token11.rds, the path to which is in $TWITTER_PAT,
#  so that next time I can do just
get_token()

#tweets_filepath <- 
previous_mla_tweets <- read_feather('data/tweets_slim_apr2019min_to_3jun2021.feather')[,c('status_id','created_at')]
existing_tweets_filepath <- 'data/tweets_full_4jun2021_to_present.feather'
existing_mla_tweets <- read_feather(existing_tweets_filepath)
cat(sprintf('Have %s tweets currently\n', prettyNum(nrow(existing_mla_tweets) + nrow(previous_mla_tweets), big.mark=',')))
#go_back_to_time <- as.POSIXct(Sys.Date()-8)  #as.character(Sys.Date()-7),'%Y-%m-%d', tz='UTC')
go_back_to_time <- max(existing_mla_tweets$created_at)
#new_mla_tweets <- get_all_tweets_back_to_date(mla_ids, go_back_to_time)
#can maybe just do in one go if guaranteeing noone will have more than 1000 (in a week)
#  About 600 should be enough to cover Aiken, Long
cat('Get Twitter users 1-40\n')
new_mla_tweets <- tweets_data(get_timelines(user = twitter_ids$user_id[1:40], n = 600))  #get all at once
cat('Get Twitter users 41-70\n')
new_mla_tweets <- rbind(new_mla_tweets, tweets_data(get_timelines(user = twitter_ids$user_id[41:70], n = 600)))
cat('Get Twitter users 71-end\n')
new_mla_tweets <- rbind(new_mla_tweets, tweets_data(get_timelines(user = twitter_ids$user_id[71:length(twitter_ids$user_id)], n = 600)))
new_mla_tweets <- new_mla_tweets[new_mla_tweets$created_at >= go_back_to_time, ]
new_mla_tweets <- new_mla_tweets[!duplicated(new_mla_tweets), ]
new_mla_tweets <- new_mla_tweets[, c('user_id','status_id','created_at','screen_name',
                             'text','source','display_text_width','reply_to_status_id',
                             'reply_to_user_id','is_quote','is_retweet','favorite_count',
                             'retweet_count','hashtags','mentions_user_id','mentions_screen_name',
                             'quoted_status_id','retweet_created_at','retweet_user_id',
                             'retweet_screen_name','retweet_description')]
new_mla_tweets$text <- plain_tweets(new_mla_tweets$text)

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
write_feather(all_mla_tweets, existing_tweets_filepath)
#and slimmer file for upload
message('Writing slim tweets file')
write_feather(all_mla_tweets[,c('user_id','screen_name','status_id','created_at','created_ym','text','is_retweet','quoted_status_id','retweet_count')], 
              str_replace(existing_tweets_filepath,'full','slim'))

#Tweet network plot

#W Humphrey user_id in data is 913773986953203714 but in twitter ids file is 913773986953203712
#K Buchanan user_id in data is 705760695812530177 but in twitter ids file is 705760695812530176
#R Butler user id in data is 1012070847501291521 but in twitter ids file is 1012070847501291520
#C Hunter 1087697152372023297 vs 1087697152372023296
#I search twitter using the latter user id but it must return the former, which an alias or replacement
#-> update the twitter ids file? or correct the tweets data as it is received?
updated_twitter_ids <- twitter_ids
updated_twitter_ids$user_id <- as.character(updated_twitter_ids$user_id)
updated_twitter_ids$user_id[updated_twitter_ids$user_id=='913773986953203712'] <- '913773986953203714'
updated_twitter_ids$user_id[updated_twitter_ids$user_id=='705760695812530176'] <- '705760695812530177'
updated_twitter_ids$user_id[updated_twitter_ids$user_id=='1012070847501291520'] <- '1012070847501291521'
updated_twitter_ids$user_id[updated_twitter_ids$user_id=='1087697152372023296'] <- '1087697152372023297'

#Need to filter by time here, rather than retweet_name_pairs, as all_mla_tweets_for_plot is used again later.
#Using last 3 months
all_mla_tweets_for_plot <- all_mla_tweets %>% 
    filter(created_at >= Sys.time() - 92*24*3600) %>%
    select(status_id, user_id, screen_name, created_ym, retweet_user_id) %>% 
    filter(!duplicated(.$status_id)) %>%
    inner_join(updated_twitter_ids %>% select(user_id, normal_name, mla_party),
               by=c('user_id')) %>%
    semi_join(politicians %>% filter(active==1), by='normal_name')
all_mla_tweets_for_plot$mla_party[all_mla_tweets_for_plot$mla_party=='Sinn Féin'] <- 'Sinn Fein'
all_mla_tweets_for_plot$mla_party[all_mla_tweets_for_plot$mla_party=='Social Democratic and Labour Party'] <- 'SDLP'
all_mla_tweets_for_plot$mla_party[all_mla_tweets_for_plot$mla_party=='Traditional Unionist Voice'] <- 'TUV'
all_mla_tweets_for_plot$mla_party[all_mla_tweets_for_plot$mla_party=='People Before Profit Alliance'] <- 'PBPA'

retweet_name_pairs <- all_mla_tweets_for_plot %>%
    filter(!is.na(retweet_user_id)) %>% 
    count(normal_name, retweet_user_id) %>% 
    inner_join(updated_twitter_ids %>% select(user_id, retweet_normal_name = normal_name),
               by=c('retweet_user_id'='user_id')) %>%
    filter(normal_name != retweet_normal_name) %>% 
    filter(retweet_normal_name %in% subset(politicians, active==1)$normal_name)
#retweeter must be in the mla list, i.e. exclude the inactive people (could change)

combined_node_list <- unique(c(retweet_name_pairs$retweet_normal_name, retweet_name_pairs$normal_name))
nodes <- data.frame(id = seq_along(combined_node_list),
                    label = combined_node_list)
edges <- retweet_name_pairs %>% inner_join(nodes, by=c('normal_name'='label')) %>% rename(from = id) %>%
    inner_join(nodes, by=c('retweet_normal_name'='label')) %>% rename(to = id) %>%
    select(from, to, weight=n)
retweets_network <- network(edges, vertex.attr = nodes, matrix.type = "edgelist", ignore.eval = FALSE,
                          directed = TRUE)  #not sure this is doing anything
network::set.vertex.attribute(retweets_network, 'party', 
                     nodes %>% left_join(unique(all_mla_tweets_for_plot[,c('normal_name','mla_party')]), by=c('label'='normal_name')) %>%
                         pull(mla_party))
#total retweets will be the sum of the 'to' column in edges
tot_retweets_by_node <- nodes %>% left_join(edges %>% group_by(to) %>% summarise(tot_retweets = sum(weight), .groups='drop'), 
                                            by=c('id'='to')) %>% 
    mutate(tot_retweets = ifelse(is.na(tot_retweets), 0, tot_retweets))
network::set.vertex.attribute(retweets_network, 'tot_retweets', tot_retweets_by_node$tot_retweets)
#power used here is important to give non-linear weighting of line thickness
#set.edge.attribute(retweets_network, 'weight_scaled',
#                   (get.edge.attribute(retweets_network,'weight') / max(get.edge.attribute(retweets_network,'weight')))^0.5)
#this is a way to be able to label only some nodes with the name rather than id
#top_10_names_by_retweets <- (arrange(tot_retweets_by_node, -tot_retweets) %>% head(10) %>% pull(label))
#set.vertex.attribute(retweets_network, 'selected_label',
#                     ifelse(get.vertex.attribute(retweets_network, 'label') %in% top_10_names_by_retweets,
#                            get.vertex.attribute(retweets_network, 'label'), NA))

#palette accepts named list, which we have in party_colours
myplot <- ggnet2(retweets_network, color='party', #palette=party_colours,
                 node.size = 'tot_retweets'#,
                 #edge.size = 'weight_scaled'
) 
myplot$data <- left_join(myplot$data, nodes %>% select(id,plot_label_all=label), by=c('label'='id'))
#myplot$data$plot_label <- ifelse(myplot$data$plot_label_all %in% top_10_names_by_retweets,
#                                 myplot$data$plot_label_all, NA)

#myplot + 
#    geom_label(aes(label = plot_label), size=3, alpha=0.5) +
#    guides(size = FALSE, colour=guide_legend(title = '', 
#                                             override.aes = list(size = 4),
#                                             label.theme = element_text(size=10)))

#get x and y on right scale for Vega
tmp <- myplot$data
tmp$x <- tmp$x * 650
tmp$y <- tmp$y * 500
#write_json(tmp, 'flask/static/tweets_network_since1july2020_nodes.json')
message('Writing tweets network files')
write_json(tmp, 'flask/static/tweets_network_last3months_nodes.json')
tmp <- edges
names(tmp) <- c('source','target','value')
#write.csv(tmp, 'flask/static/tweets_network_since1july2020_edges.csv', quote=FALSE, row.names=FALSE)
write.csv(tmp, 'flask/static/tweets_network_last3months_edges.csv', quote=FALSE, row.names=FALSE)

#Get top 5s of network measures
myplot$data$betw_centr <- sna::betweenness(retweets_network, cmode='directed')
myplot$data$info_centr <- sna::infocent(retweets_network) 
library(igraph)
library(intergraph)
myplot$data$page_rank <- igraph::page_rank(asIgraph(retweets_network))$vector
data.frame(rank=seq(5), 
           info_centr = arrange(myplot$data, -info_centr) %>% head(5) %>% pull(plot_label_all),
           page_rank = arrange(myplot$data, -page_rank) %>% head(5) %>% pull(plot_label_all),
           betw_centr = arrange(myplot$data, -betw_centr) %>% head(5) %>% pull(plot_label_all)) %>%
    write.csv('data/tweets_network_last3months_top5s.csv', quote=FALSE, row.names=FALSE)

rm(all_mla_tweets, new_mla_tweets, existing_mla_tweets, all_mla_tweets_for_plot, tmp)
