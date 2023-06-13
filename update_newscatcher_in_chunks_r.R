#Script to update the news file while operating within the limits of NewsCatcher API
#- 21 requests per hour
#- max 20 pages per request, so have to break up in time if expect more than 100 hits
#  - Foster 1 week can be ~40 pages, O'Neill 10-20
#  - Stewart 5-10 because mostly false positives
#  - others are OK (Murphy,Long,Mallon ~3-5 pages per week)
#-> Run full set weekly, and do midweek extra run for the big ones
# Run before the other weekly update scripts, and once midweek if necessary. Then EC2 instance only
#   needs to be up for ~6 or 12 hours a week.

#Don't really get any false positives with name in quotes except for a few like Allen, Stewart, especially if use source selection.
#So only apply the content filtering to those people.

suppressPackageStartupMessages(library(dplyr))
library(httr)
library(jsonlite)
suppressPackageStartupMessages(library(feather))

if (Sys.info()['user']=='rstudio') setwd('/home/rstudio/nipol')

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

filter_newscatcher_hits <- function(hits_articles, mla_name) {
    
    hits <- hits_articles %>% rename(source = clean_url) %>%
        mutate(normal_name = mla_name) %>%
        select(normal_name, published_date, title, link, source, summary) %>% 
        #filter(grepl('Northern Ireland|NI|Belfast|NI Assembly|Northern Ireland Assembly|NI Executive|Northern Ireland Executive|First Minister|Finance Minister|Health Minister|Environment Minister|Education Minister|MLA|\\bMP\\b|UUP|Ulster Unionist|SDLP|Social Democratic|DUP|Democratic Unionist|Alliance|Sinn F|TUV|Traditional Unionist|PBP|People Before Profit', summary),
        filter(!grepl('\\.mp3', link))  #rasset.ie seem to be podcasts
    #stricter filter in some cases
    if (mla_name %in% c('John Stewart','Andy Allen','John Blair','Jim Wells','Simon Hamilton','Conor Murphy','Rosemary Barton')) {
        hits <- hits %>% 
            filter(grepl('Northern Ireland|NI|Belfast|NI Assembly|Northern Ireland Assembly|NI Executive|Northern Ireland Executive|First Minister|Finance Minister|Health Minister|Environment Minister|Education Minister|MLA|\\bMP\\b|UUP|Ulster Unionist|SDLP|Social Democratic|DUP|Democratic Unionist|Alliance|Sinn F|TUV|Traditional Unionist|PBP|People Before Profit', summary),
                   !(grepl('Andy Allen', summary) & grepl('masterchef', tolower(summary))))
    }
    
    #dedup where articles are identical but in different domains
    hits <- hits %>% sample_frac(1) %>% 
        mutate(title = gsub('’', '\'', title)) %>%
        filter(!duplicated(.[,c('normal_name', 'title')])) %>% arrange(published_date)
    hits
}

do_21_news_gets <- function(politicians_normal_names, politician_start_index, existing_news_articles,
                            start_page_num=1) {

    cat('index at start of function =',politician_start_index,'\n')
    sources_to_check <- c("belfasttelegraph.co.uk", "bbc.co.uk", "irishtimes.com", "thejournal.ie",
                          "independent.ie", "agriland.ie", "rte.ie", "belfastlive.co.uk",
                          "highlandradio.com", "thesun.ie", "metro.co.uk", "thetimes.co.uk",
                          "telegraph.co.uk", "breakingnews.ie", "theguardian.com",
                          "the42.ie", "qub.ac.uk", "express.co.uk", "yahoo.com",
                          "thesun.co.uk", "mirror.co.uk", "limerickleader.ie", "irishcentral.com",
                          "inews.co.uk", "huffingtonpost.co.uk", "google.com", 
                          "dailystar.co.uk", "dailymail.co.uk", "reuters.com", "independent.co.uk",
                          "thestandard.com.hk", "kfgo.com", "catholicworldreport.com",
                          "thenational.scot", "thelondoneconomic.com", "walesonline.co.uk",
                          "standard.co.uk", "irishmirror.ie", "pioneergroup.com", "metro.us",
                          "birminghammail.co.uk","rt.com","forexlive.com",
                          "donegaldaily.com", "dailyrecord.co.uk", "manchestereveningnews.co.uk",
                          "newstatesman.com", "thedailybeast.com", "thescottishsun.co.uk")
    #New ones:
    #sluggerotoole.com? Not really news
    #politico.eu
    
    collected_new_results <- data.frame()
    gets_in_this_chunk <- 0
    got_429 <- FALSE
    next_time_start_page_num <- 1
    
    #will get through at most 21 politicians
    for (p in seq(politician_start_index, min(politician_start_index+20, length(politicians_normal_names)))) {

        mla_name <- politicians_normal_names[p]
        finished_this_mla <- FALSE
        
        #find date to go back to - possibly overlap with most recent date by 1 day
        from_date <- as.character(Sys.Date()-8)
        #if (sum(existing_news_articles$normal_name==mla_name) > 0) {
        #    from_date <- max(from_date, 
        #                     substr(max(subset(existing_news_articles, normal_name==mla_name)$published_date),1,10))
        #}
        to_date <- Sys.Date()+1
        
        search_mla_name <- prepare_search_mla_name(mla_name)
        
        page_num <- if (gets_in_this_chunk==0) start_page_num else 1
        cat(sprintf('Starting with page %i\n', page_num))
        total_pages <- 999
        new_rows <- data.frame()
        while (page_num <= total_pages & page_num <= 20) {
            
            tmp <- GET(sprintf('https://newscatcher.p.rapidapi.com/v1/search?lang=en&page_size=5&page=%i&q="%s"&from=%s&to=%s&sources=%s',
                               page_num, gsub(' ','%20',search_mla_name), from_date, to_date, paste(sources_to_check, collapse=',')),
                       add_headers('x-rapidapi-host' = 'newscatcher.p.rapidapi.com',
                                   'x-rapidapi-key' = newscatcher_api_key))
            cat(sprintf('%g %s: page %g of %g\n',p, mla_name, page_num, total_pages))
            
            if (tmp$status_code == '429') {
                print('Hit hourly limit')
                got_429 <- TRUE
                break
            } else {
                hits <- fromJSON(content(tmp, as='text', encoding = 'utf8'))
            }
        
            if ('articles' %in% names(hits)) {
                total_pages <- hits$total_pages  #update from the initial guess of 1
                new_rows <- rbind(new_rows, filter_newscatcher_hits(hits$articles, mla_name))
            } else if (total_pages == 999) {
                total_pages <- 0
            }
            
	        #we can't easily go past 20 pages so call it done
            if (page_num >= total_pages | page_num == 20) {
                finished_this_mla <- TRUE
                politician_start_index <- p+1
                collected_new_results <- rbind(collected_new_results, new_rows)
            }
            
            page_num <- page_num + 1
            
            gets_in_this_chunk <- gets_in_this_chunk + 1
            if (gets_in_this_chunk >= 21) {
                collected_new_results <- rbind(collected_new_results, new_rows)
                cat('Done for now\n')
                cat('Have',nrow(collected_new_results),'possibly new rows\n')
                next_time_start_page_num <- if (page_num <= total_pages) page_num else 1
                cat(sprintf('Next time start at politician index %i and page %i\n', politician_start_index, page_num))
                break
            }
        }
        cat('gets_in_this_chunk =',gets_in_this_chunk,'\n')
        if (gets_in_this_chunk >= 21 | got_429) break
    }

    #For the case that we have got to the end of politicians_normal_names without needing to break:
    if (gets_in_this_chunk < 21 & !got_429) politician_start_index <- -999

    #return index to start at next time
    list('collected_new_results'=collected_new_results, 
         'next_time_start_index'=politician_start_index,
         'next_time_start_page_num'=next_time_start_page_num)
}

#Run
existing_news_articles_file_name <- 'data/newscatcher_articles_sep2020topresent.feather'
news_update_place_file_name <- 'newscatcher_politician_index_to_start_at.txt'

existing_news_articles <- read_feather(existing_news_articles_file_name)

politicians <- read.csv('data/all_politicians_list.csv', stringsAsFactors=FALSE)
newscatcher_api_key <- readLines('newscatcher_api_token', n=1)

search_type <- 'weekly'

#TODO check the start index bit for midweek
if (search_type == 'midweek') {
    #Can get up to 100 at a time (5*20, if no false positives), so anybody producing more than this
    #  in a week needs to be queried another time too
    politicians_normal_names_to_check <- existing_news_articles %>% mutate(year=year(published_date),
                                                                 weeknum=week(published_date)) %>%
        filter(year==max(year)) %>% count(weeknum,normal_name) %>% filter(n >= 40) %>% 
        pull(normal_name) %>% unique()
    
    cat(sprintf('Doing midweek query; checking %s only\n', paste(politicians_normal_names_to_check, collapse=', ')))
} else {
    politicians_normal_names_to_check <- politicians$normal_name
}


#Full list for recent 3-7 days should take ~1 get for most people and 2-10 for the big ones 
#  so maybe 1.2*90 + 5*10 ~ 170 gets at most = 8 times in loop, which will take 7*gap ~ 4 hours
next_time_start_page_num <- 1
for (i in seq(100)) {
    politician_start_index <- as.integer(readLines(news_update_place_file_name))
    
    cat('Starting chunk at',politician_start_index,'\n')
    tmp <- do_21_news_gets(politicians_normal_names_to_check, politician_start_index, existing_news_articles,
                           start_page_num = next_time_start_page_num)
    next_time_start_index <- tmp$next_time_start_index
    next_time_start_page_num <- tmp$next_time_start_page_num
    collected_new_results <- tmp$collected_new_results
    cat(sprintf('After 21 gets, next_time_start_index = %i\n', next_time_start_index))
    cat(sprintf('Next time start at page_num = %i\n', next_time_start_page_num))

    #Append collected_new_results, which has some complete politicians, to file
    #Dedup after appending because we might have got articles from most recent day for a second time
    prev_nrows <- nrow(existing_news_articles)
    existing_news_articles <- rbind(existing_news_articles, collected_new_results)
    #existing_news_articles <- existing_news_articles[!duplicated(existing_news_articles), ]
    existing_news_articles$short_date <- as.Date(existing_news_articles$published_date)
    existing_news_articles <- existing_news_articles[!duplicated(existing_news_articles %>% select(normal_name, link, short_date)), ]
    existing_news_articles$short_date <- NULL
    #
    cat(sprintf('Added %i rows to file\n', nrow(existing_news_articles)-prev_nrows))
    write_feather(existing_news_articles, existing_news_articles_file_name)
    
    #Update place file for next time
    writeLines(as.character(next_time_start_index), news_update_place_file_name)
    
    if (next_time_start_index == -999 | next_time_start_index > length(politicians_normal_names_to_check)) {
        writeLines('1', news_update_place_file_name)
        cat('Done; breaking\n')
        break
    } else {
        cat('Waiting for a while \n')
        Sys.sleep(60*16) #maybe 15-20 minutes is enough
    }
}
