suppressPackageStartupMessages(library(dplyr))
library(httr)
library(jsonlite)
library(feather)
suppressPackageStartupMessages(library(rtweet))

if (Sys.info()['user']=='rstudio') setwd('/home/rstudio/nipol')

existing_news_articles_file_name <- 'data/newscatcher_articles_sep2020topresent.feather'
existing_news_articles <- read_feather(existing_news_articles_file_name)

politicians <- read.csv('data/all_politicians_list.csv', stringsAsFactors=FALSE)


#Get Irish News from Twitter? They may Tweet the first line of every article.
if (Sys.info()['user']=='rstudio') Sys.setenv('TWITTER_PAT'='/home/rstudio/nipol/.rtweet_token11.rds')
get_token()

#irish_news, user_id 99960420
#tmp <- tweets_data(get_timelines(user = 99960420, n = 200))
#table(mday(tmp$created_at))  #200 covers ~5 days
#Some are retweets of their journalists - can check for irishnews.com in the urls_expanded_url
#Use the url for the title and the text for the first line, i.e.
#tmp$text[3], and
#strsplit(tmp$urls_expanded_url[[3]],'/')[[1]] %>% .[[length(.)]] %>% strsplit('-') %>% unlist() %>% paste(collapse=' ')

get_article_title_from_hyphenated_url <- function(url) {
    if (grepl('trib\\.al', url) & nchar(url) <= 35) {
        return('-')
    } else {
        lower_case_string <- sub('\\?param\\=[^\\s]*$', '', url, perl=TRUE)
        lower_case_string <- strsplit(lower_case_string,'/')[[1]] %>% .[[length(.)]] %>% strsplit('-') %>% unlist()

        #maybe drop a story id from the end
        if (grepl('^\\d*$', lower_case_string[length(lower_case_string)])) {
            lower_case_string <- lower_case_string[1:(length(lower_case_string)-1)]
        }
        #and a url
        lower_case_string <- sub('\n*http[^\\s]*$', '', lower_case_string, perl=TRUE)

        lower_case_string <- trimws(paste(lower_case_string, collapse=' '))

        return(paste(toupper(substring(lower_case_string, 1, 1)), substring(lower_case_string, 2), sep = "", collapse = " "))
    }
}

recent_irish_news_tweets <- tweets_data(get_timelines(user = 99960420, n = 500)) %>% filter(!is_retweet)
recent_irish_news_tweets$published_date <- as.character(recent_irish_news_tweets$created_at)
recent_irish_news_tweets$link <- sapply(recent_irish_news_tweets$urls_expanded_url, function(x) x[[1]])
recent_irish_news_tweets$title <- as.character(sapply(recent_irish_news_tweets$link, get_article_title_from_hyphenated_url))
recent_irish_news_tweets$source <- 'irishnews.com'
recent_irish_news_tweets$summary <- as.character(sapply(recent_irish_news_tweets$text, function(t) sub('\n*http[^\\s]*$', '', t, perl=TRUE)))

#And same for News Letter, 135920727
#tmp <- tweets_data(get_timelines(user = 135920727, n = 200))
#table(mday(tmp$created_at))  #200 covers ~8 days
#Some urls have the title but about half are trib.al/* with no title; set title='-' for these

recent_news_letter_tweets <- tweets_data(get_timelines(user = 135920727, n = 300)) %>% filter(!is_retweet)
recent_news_letter_tweets$published_date <- as.character(recent_news_letter_tweets$created_at)
recent_news_letter_tweets$link <- sapply(recent_news_letter_tweets$urls_expanded_url, function(x) x[[1]])
recent_news_letter_tweets$title <- as.character(sapply(recent_news_letter_tweets$link, get_article_title_from_hyphenated_url))
recent_news_letter_tweets$source <- 'newsletter.co.uk'
recent_news_letter_tweets$summary <- as.character(sapply(recent_news_letter_tweets$text, function(t) sub('\n*http[^\\s]*$', '', t, perl=TRUE)))

#Dedup by link only, before adding to the main file (haven't duplicated by involving person yet), 
#  to avoid repeated tweets about the same article quoting a different line each time
recent_irish_news_tweets <- recent_irish_news_tweets[!duplicated(recent_irish_news_tweets$link),]
recent_news_letter_tweets <- recent_news_letter_tweets[!duplicated(recent_news_letter_tweets$link),]

#To save, we need published_date, title, link, source, summary, and normal_name for each match
inews_or_nl_matches <- data.frame()
#Use lower case to search, because I lower-cased most of the title
for (politician in politicians$normal_name) {
    inews_or_nl_matches <- rbind(inews_or_nl_matches,
                                 recent_irish_news_tweets %>% filter(grepl(tolower(politician), tolower(title)) | grepl(tolower(politician), tolower(summary))) %>%
                                     mutate(normal_name = politician) %>%
                                     select(normal_name, published_date, title,
                                            link, source, summary))
    inews_or_nl_matches <- rbind(inews_or_nl_matches,
                                 recent_news_letter_tweets %>% filter(grepl(tolower(politician), tolower(title)) | grepl(tolower(politician), tolower(summary))) %>%
                                     mutate(normal_name = politician) %>%
                                     select(normal_name, published_date, title,
                                            link, source, summary))
}
#table(inews_or_nl_matches$normal_name)

prev_nrows <- nrow(existing_news_articles)
existing_news_articles <- rbind(existing_news_articles, inews_or_nl_matches)
#dedup - don't allow more than one normal_name-link per day
existing_news_articles$short_date <- as.Date(existing_news_articles$published_date)
existing_news_articles <- existing_news_articles[!duplicated(existing_news_articles %>% select(normal_name, link, short_date)), ]
existing_news_articles$short_date <- NULL
#
cat(sprintf('Added %i rows to file\n', nrow(existing_news_articles)-prev_nrows))
write_feather(existing_news_articles, existing_news_articles_file_name)
