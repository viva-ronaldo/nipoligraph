# Update news source using NewsCatcher API V2 (valid for at least Nov 23-Jan 24)

suppressPackageStartupMessages(library(dplyr))
library(httr)
library(jsonlite)
suppressPackageStartupMessages(library(arrow))
library(stringr)

if (Sys.info()['user']=='rstudio') setwd('/home/rstudio/nipol')

data_dir = './data'

prepare_search_mla_name <- function(mla_name) {
    search_mla_name <- mla_name
    if (mla_name == 'Ian Paisley Jr') search_mla_name <- 'Ian Paisley'  #should give a more accurate total articles
    
    search_mla_name <- tolower(search_mla_name)
    
    #can't use irish characters in GET
    search_mla_name <- gsub('é','e', search_mla_name)
    search_mla_name <- gsub('ó','o', search_mla_name)
    search_mla_name <- gsub('á','a', search_mla_name)
    search_mla_name <- gsub('í','i', search_mla_name)
    search_mla_name
}

assign_newscatcher_hits_to_politician <- function(hits_articles, politicians,
    url_patterns_to_exclude, article_content_to_remove) {
    
    clean_articles <- hits_articles %>%
        rename(source = clean_url) %>%
        select(published_date, title, link, source, summary, is_opinion) %>%
        mutate(summary = str_squish(summary),
               lc_summary = tolower(summary))

    # Exclude url patterns that are likely to be false positives
    clean_articles <- filter(clean_articles, !grepl(paste(url_patterns_to_exclude, collapse='|'), link))

    # Exclude opinion articles
    clean_articles <- subset(clean_articles, !is_opinion)

    # Filter to NI relevant - safer to do this; doesn't give many false negatives
    clean_articles <- clean_articles %>% filter(grepl('Northern Ireland|NI|Belfast|NI Assembly|Northern Ireland Assembly|NI Executive|Northern Ireland Executive|First Minister|Finance Minister|Minister of Finance|Economy Minister|Minister for the Economy|Health Minister|Minister of Health|Environment Minister|Agriculture Minister|Minister of Agriculture|Infrastructure Minister|Minister for Infrastructure|Education Minister|Minister of Education|Justice Minister|Minister of Justice|Communities Minister|Minister for Communities|MLA|\\bMP\\b|UUP|Ulster Unionist|SDLP|Social Democratic|DUP|Democratic Unionist|Alliance|Sinn F|TUV|Traditional Unionist|PBP|People Before Profit|PUP|Progressive Unionist|Aontu', summary))

    # Remove some phrases that aren't part of the article        
    clean_articles <- mutate(clean_articles, summary = str_squish(
        str_remove_all(summary, paste(article_content_to_remove, collapse='|'))))
    
    hits <- data.frame()
    for (p in unique(politicians$normal_name)) {
        p_hits <- clean_articles %>%
            filter(grepl(prepare_search_mla_name(p), lc_summary)) %>%
            mutate(normal_name = p)
        
        hits <- rbind(hits, p_hits)
    }
    
    hits <- hits %>% select(normal_name, published_date, title, link, source, summary)
    
    #dedup where articles (titles) are identical but in different domains
    hits <- hits %>% sample_frac(1) %>% 
        mutate(title = gsub('’', '\'', title)) %>%
        filter(!duplicated(.[,c('normal_name', 'title')])) %>% arrange(published_date)
    hits
}

sources_to_check <- c("agriland.ie",
                      "bbc.co.uk",
                      "belfastlive.co.uk",
                      "belfasttelegraph.co.uk",
                      "birminghammail.co.uk",
                      "breakingnews.ie",
                      "catholicworldreport.com",
                      "dailymail.co.uk",
                      "dailyrecord.co.uk",
                      "dailystar.co.uk",
                      "donegaldaily.com",
                      "express.co.uk",
                      "google.com",
                      "highlandradio.com",
                      "huffingtonpost.co.uk",
                      "independent.co.uk",
                      "independent.ie",
                      "inews.co.uk",
                      "irishcentral.com",
                      "irishmirror.ie",
                      "irishtimes.com",
                      "limerickleader.ie",
                      "kfgo.com",
                      "manchestereveningnews.co.uk",
                      "metro.co.uk",
                      "metro.us",
                      "mirror.co.uk",
                      "newstatesman.com",
                      "pioneergroup.com",
                      "qub.ac.uk",
                      "reuters.com",
                      "rt.com",
                      "rte.ie",
                      "standard.co.uk",
                      "telegraph.co.uk",
                      "the42.ie",
                      "thedailybeast.com",
                      "theguardian.com",
                      "thejournal.ie",
                      "thelondoneconomic.com",
                      "thenational.scot",
                      "thescottishsun.co.uk",
                      "thestandard.com.hk",
                      "thesun.co.uk",
                      "thesun.ie",
                      "thetimes.co.uk",
                      "walesonline.co.uk",
                      "newsletter.co.uk",
                      "irishnews.com"
                      )

# Some paths within these can be filtered out as likely false positives; >50% are false positives
url_patterns_to_exclude <- c("/sport",
                             "/football",
                             "/culture/books",
                             "/tv",
                             "/showbiz",
                             "/arts"
                             )
# keep /culture, /entertainment, /health, /lifestyle - some of these are real, and article content filter later will be used

article_content_to_remove <- c("Read more: ?",
                               "READ MORE: ?",
                               "Advertisement Hide Ad ?",
                               "Learn More ?",
                               "Sorry, there seem to be some issues. Please try again later. Submitting...",
                               "Sign up Thank you for signing up! Did you know with a Digital Subscription to Belfast News Letter, you can get unlimited access to the website including our premium content, as well as benefiting from fewer ads, loyalty rewards and much more. ?",
                               "Don't miss the latest news from around Scotland and beyond - Sign up to our daily newsletter here ?",
                               "Something went wrong, please try again later. ?",
                               "Get daily headlines and breaking news alerts for FREE by signing up to our newsletter ?",
                               "Stay on top of the headlines from Belfast and beyond by signing up for FREE email alerts ?",
                               "The video will auto-play soon 8 Cancel Click to play Tap to play ?",
                               "See our privacy notice ?",
                               "We have more newsletters ?",
                               "Sign up! ?",
                               "Thank you for subscribing!? ?",
                               "Want the biggest political stories and analysis sent to you every week\\? Sign up to our FREE newsletter ",
                               "Get the latest nostalgia features and photo stories from Belfast Nostalgia straight to your inbox ",
                               "Invalid [eE]mail ?",
                               "Get the latest news from across Ireland straight to your inbox every single day ?",
                               "Join the Irish Mirror's breaking news service on WhatsApp. ?",
                               "Click this link to receive breaking news and the latest headlines direct to your phone. ?",
                               "We also treat our community members to special offers, promotions, and adverts from us and our partners. ?",
                               "If you don't like our community, you can check out any time you like. ?",
                               "If you're curious, you can read our Privacy Notice.",
                               "Watch more of our videos on Shots! and live on Freeview channel 276 Visit Shots! now ",
                               "News Sport Business Lifestyle Culture Going Out Homes & Property Comment",
                               "Sort by Oldest first Newest first Highest scored Lowest scored",
                               "We know there are thousands of National readers who want to debate, argue and go back and forth in the comments section of our stories. We've got the most informed readers in Scotland, asking each other the big questions about the future of our country. Unfortunately, though, these important debates are being spoiled by a vocal minority of trolls who aren't really interested in the issues, try to derail the conversations, register under fake names, and post vile abuse. So that's why we've decided to make the ability to comment only available to our paying subscribers. That way, all the trolls who post abuse on our website will have to pay if they want to join the debate – and risk a permanent ban from the account that they subscribe with. The conversation will go back to what it should be about – people who care passionately about the issues, but disagree constructively on what we should do about them. Let's get that debate started! Callum Baird, Editor of The National ?",
                               "Join our Belfast Live breaking news service on WhatsApp ?",
                               "Click this link or scan the QR code to receive breaking news and top stories from Belfast Live. ?",
                               "For all the latest news, visit the Belfast Live homepage here and sign up to our daily newsletter here. ?"
                               )

# Bi-weekly run
message('\nDoing NewsCatcher update')
existing_news_articles_filepath <- file.path(data_dir, 'newscatcher_articles_sep2020topresent.feather')
existing_news_articles <- read_feather(existing_news_articles_filepath)

politicians <- read.csv(file.path(data_dir, 'all_politicians_list.csv'), stringsAsFactors=FALSE)
search_politician_names <- as.character(sapply(politicians$normal_name, prepare_search_mla_name))
search_politician_names <- search_politician_names[!duplicated(search_politician_names)]

newscatcher_api_key <- readLines('newscatcher_api_token', n=1)

page_num <- 1
total_pages <- 999
new_rows <- data.frame()
while (page_num <= total_pages & page_num <= 20) {
    tmp <- GET(sprintf('https://api.newscatcherapi.com/v2/search?lang=en&page_size=50&page=%i&from=%s&q="%s"&sources=%s',
                       page_num,
                       gsub(' ', '%20', '4 days ago'),
                       gsub(' ', '%20', paste(search_politician_names, collapse='"OR"')),
                       paste(sources_to_check, collapse=',')),
               add_headers('x-api-key' = newscatcher_api_key))
    
    if (tmp$status_code != '200') {
        message('Error with NewsCatcher request; stopping')
        break
    } else {
        hits <- fromJSON(content(tmp, as='text', encoding = 'utf8'))
    }
    
    if ('articles' %in% names(hits)) {
        total_pages <- hits$total_pages  #update from the initial guess of 1
        new_rows <- rbind(new_rows, assign_newscatcher_hits_to_politician(
            hits$articles, politicians, url_patterns_to_exclude, article_content_to_remove))
    } else if (total_pages == 999) {
        total_pages <- 0
    }
    
    page_num <- page_num + 1
}
message(sprintf('- Retrieved %i pages from API containing %i articles', page_num-1, nrow(new_rows)))
new_rows <- arrange(new_rows, published_date, normal_name)

#Append new hits to file
#Dedup after appending because we might have got articles from most recent day for a second time
prev_nrows <- nrow(existing_news_articles)
existing_news_articles <- rbind(existing_news_articles, new_rows) %>%
    mutate(short_date = as.Date(published_date)) %>%
    filter(!duplicated(.[, c('normal_name', 'link', 'short_date')])) %>%
    select(-short_date)
#
write_feather(existing_news_articles, existing_news_articles_filepath)
cat(sprintf('- Added %i rows to news article file\n', nrow(existing_news_articles)-prev_nrows))


# Notes
# # Default 'from' is 1 week ago.
# tmp <- GET(sprintf('https://api.newscatcherapi.com/v2/search?lang=en&page_size=50&from=%s&q="%s"&sources=%s',
#                    gsub(' ', '%20', '4 days ago'),
#                    gsub(' ', '%20', paste(search_politician_names, collapse='"OR"')),
#                    #gsub(' ','%20',search_mla_name),
#                    paste(sources_to_check, collapse=',')),
#            add_headers('x-api-key' = newscatcher_api_key))

# #tmp$status_code  # 200
# hits <- fromJSON(content(tmp, as='text', encoding = 'utf8'))
# #hits$total_hits  # 203
# #hits$total_pages # 14
# hits$articles   # df, 19 columns, page_size rows
# # use same columns as before: published_date, title, link, summary; 'clean_url' instead of 'source'.
# # Also have 'excerpt'; 'topic' = 'politics' or 'news'; 'is_opinion' T/F; 'score' - higher means better query match.
