#Functions to update various parts of the data

#Assembly answers to a list of pending questions
get_answers_to_questions <- function(documentId_list) {
    answers_to_questions <- data.frame()
    for (q_id in unique(documentId_list)) {
        #cat(q_id,'\n')
        tmp <- GET(sprintf('http://data.niassembly.gov.uk/questions.asmx/GetQuestionDetails_JSON?documentId=%s', q_id))
        tmp <- fromJSON(content(tmp, as='text'))$QuestionsList$Question
        
        if (!is.data.frame(tmp)) {
            #occasionally need to handle a NULL here
            tmp <- lapply(tmp, function(x) ifelse(is.null(x), NA, x))
            tmp <- data.frame(tmp)
        }
        if (nrow(tmp) == 0) next
        
        #Can get duplicates- looks like where one answer should be appended to another
        #OR can be minister and deputy first minister getting a row each as MinisterPersonId - don't concat these
        if (nrow(tmp) > 1 && 
            (('MinisterPersonId' %in% names(tmp) && n_distinct(tmp$MinisterPersonId) == 1) ||
             (tmp$MinisterTitle == 'Assembly Commission'))) {
            full_answer <- paste(tmp$AnswerPlainText, sep=' _PLUS_ ')
            tmp <- tmp[order(tmp$AnsweredOnDate), ]
            tmp <- tmp[1,]
            cat(sprintf('concatenated answers for document id %s\n', tmp$DocumentId[1]))
        }
        
        #Not all have a MinisterPersonId - they are Ministertitle == Assembly Commission
        if (!('MinisterPersonId' %in% names(tmp))) tmp$MinisterPersonId <- NA
        
        #If AnswerByDate is present instead of AnsweredOnDate, it hasn't been answered yet
        if ('AnsweredOnDate' %in% names(tmp)) {
            answers_to_questions <- rbind(answers_to_questions, 
                                          tmp %>% select(DocumentId, DocumentType, TablerName, TablerPersonId,
                                                         TabledDate, AnsweredOnDate, PriorityRequest,
                                                         MinisterPersonId, MinisterTitle))
        }
    }
    return(answers_to_questions)
}

#Division votes - just check what the most recent date was and go from there
update_vote_list <- function(votes_file, vote_results_file) {
    existing_votes <- read_feather(votes_file)
    existing_vote_results <- read_feather(vote_results_file)
    message('Latest existing vote date =',max(existing_votes$DivisionDate))
    
    tmp <- GET(sprintf('http://data.niassembly.gov.uk/plenary.asmx/GetVotesOnDivision_JSON?startDate=%s&endDate=%s',
                       substr(max(existing_votes$DivisionDate), 1, 10), Sys.Date()))
    possible_new_votes <- fromJSON(content(tmp, as='text'))$DivisionList$Division
    if (!is.data.frame(possible_new_votes)) {
        #got one result so it became a list
        possible_new_votes <- data.frame(possible_new_votes)
    }
    #In May 2021 hundreds of old records appeared with DivisionDate=2021-05-04
    #Handle this by requiring EventId (as integer) to be at least the oldest vote in my records, which is 11486 from 2020-01-11
    possible_new_votes <- subset(possible_new_votes, as.integer(EventID) >= min(as.integer(existing_votes$EventId)))
    #The ones >=11486 that I already have will be filtered out in the next step
    
    #The true vote date, as shown on the vote pages like http://aims.niassembly.gov.uk/plenary/details.aspx?&ses=0&doc=332585&pn=0&sid=vd,
    #  can be got from GetPlenaryDetails_JSON, as $PlenaryList$Plenary$PlenaryDate
    
    if (!is.null(possible_new_votes)) {
        if (!is.data.frame(possible_new_votes)) possible_new_votes <- data.frame(possible_new_votes)
        
        possible_new_votes <- arrange(possible_new_votes, DivisionDate) %>%
            select(EventId=EventID, SessionID, DivisionSubject, DivisionDate, DocumentID) %>%
            filter(!(EventId %in% existing_votes$EventId)) 
        
        cat('Found',nrow(possible_new_votes),'possible new votes\n')
        if (nrow(possible_new_votes) > 0) {
            
            new_vote_results <- data.frame()
            new_vote_summary <- data.frame()
            for (d_id in possible_new_votes$DocumentID) {
                #get the tabler(s)
                tmp <- GET(sprintf('http://data.niassembly.gov.uk/plenary.asmx/GetPlenaryTablers_JSON?documentid=%s', d_id))
                tabler_personIDs <- fromJSON(content(tmp, as='text'))$TablerList$Tabler
                #and the result summary
                tmp <- GET(sprintf('http://data.niassembly.gov.uk/plenary.asmx/GetDivisionResult_JSON?documentid=%s', d_id))
                division_result <- fromJSON(content(tmp, as='text'))$DivisionDetails$Division
                #and the true vote date
                tmp <- GET(sprintf('http://data.niassembly.gov.uk/plenary.asmx/GetPlenaryDetails_JSON?documentId=%s',d_id))
                vote_date <- fromJSON(content(tmp, as='text'))$Plenary$Plenary$PlenaryDate
                #save what we need from these
                new_row <- data.frame(division_result[c('EventId','Title','DecisionMethod',
                                                        'DecisionType','Outcome')])
                new_row['tabler_personIDs'] <- paste(tabler_personIDs$TablerPersonID, collapse=';')
                new_row['true_DivisionDate'] <- vote_date
                new_vote_summary <- rbind(new_vote_summary, new_row)
                
                tmp <- GET(sprintf('http://data.niassembly.gov.uk/plenary.asmx/GetDivisionMemberVoting_JSON?documentId=%s',d_id))
                tmp <- fromJSON(content(tmp, as='text'))$MemberVoting$Member
                new_vote_results <- rbind(new_vote_results, tmp[,c('EventID','PersonID','Vote','Designation')])
            }
            names(new_vote_results)[names(new_vote_results)=='EventID'] <- 'EventId'
            names(new_vote_results)[names(new_vote_results)=='PersonID'] <- 'PersonId'
            #print(new_vote_results)
            
            #replace the dodgy DivisionDate with the better PlenaryDate=true_DivisionDate
            new_vote_summary <- left_join(possible_new_votes, new_vote_summary, by='EventId') %>%
                mutate(DivisionDate=true_DivisionDate) %>% select(-true_DivisionDate)
            
            existing_votes <- rbind(existing_votes, new_vote_summary)
            existing_vote_results <- rbind(existing_vote_results, new_vote_results)
            write_feather(existing_votes, votes_file)
            write_feather(existing_vote_results, vote_results_file)
        }
    }
    invisible()
}

update_and_get_politicians_list <- function(path_to_politicians_file='data/all_politicians_list.csv') {
    politicians <- read.csv(path_to_politicians_file)
    # Sometimes get no response from address; we don't need to do the update, can fall back to using current politicians list
    tryCatch({
        tmp <- GET(sprintf('http://data.niassembly.gov.uk/members.asmx/GetAllCurrentMembers_JSON'))
        mlas <- fromJSON(content(tmp, as='text'))$AllMembersList$Member
        new_mlas <- subset(mlas, !(MemberName %in% politicians$MemberName))
    },
    error = function(e) {
        message("Couldn't reach data.niassembly.gov.uk; using existing politicians list")
        new_mlas <- data.frame()
    })
    if (nrow(new_mlas) > 0) {
        politicians <- rbind(politicians,
                             new_mlas %>% mutate(role='MLA', normal_name=paste(MemberFirstName, MemberLastName), added=as.character(Sys.Date()), active=1) %>%
                                 select(PersonId, normal_name, PartyName, role, added, active, 
                                        MemberName, MemberFirstName, MemberLastName, ConstituencyName, ConstituencyId))
        write.csv(politicians, path_to_politicians_file, row.names=FALSE)
        
        #Make a note of this change
        write.table(data.frame(d=as.character(Sys.Date()), t=sprintf('Added new MLAs %s', paste(new_mlas$MemberLastName, collapse=','))), 
                    'politician_list_changes.log', 
                    append=TRUE, quote=FALSE, sep=' ', col.names=FALSE, row.names=FALSE)
        
    }
    #Also check for any MLAs that have left and set them to inactive in the file
    newly_inactive_mlas <- subset(politicians, role=='MLA' & active==1 & !(MemberName %in% mlas$MemberName))
    if (nrow(newly_inactive_mlas) > 0) {
        politicians %<>% mutate(active = ifelse(MemberName %in% newly_inactive_mlas$MemberName, 0, active))
        write.csv(politicians, path_to_politicians_file, row.names=FALSE)
        write.table(data.frame(d=as.character(Sys.Date()), t=sprintf('Removed inactive MLAs %s', paste(newly_inactive_mlas$MemberLastName, collapse=','))), 
                    'politician_list_changes.log', 
                    append=TRUE, quote=FALSE, sep=' ', col.names=FALSE, row.names=FALSE)
    }
    
    politicians
}

update_assembly_lists <- function() {
    tryCatch({
        #Diary of business
        tmp <- GET(sprintf('http://data.niassembly.gov.uk/plenary.asmx/GetBusinessDiary_JSON?startDate=%s&endDate=2025-01-01', Sys.Date()))
        diary <- fromJSON(content(tmp, as='text'))$BusinessDiary$DiaryItem
        write.table(diary, 'data/diary_future_events.psv', quote=FALSE, row.names=FALSE, sep='|')
        
        #Latest committee memberships 
        tmp <- GET('http://data.niassembly.gov.uk/members.asmx/GetAllMemberRoles_JSON?')
        roles_list <- fromJSON(content(tmp, as='text'))$AllMembersRoles$Role
        committees_list <- subset(roles_list, Role %in% c('Committee Member','Committee Chair','Committee Deputy Chair'))
        write.csv(committees_list[,c('PersonId','Organisation','Role')], 'data/current_committee_memberships.csv', quote=TRUE, row.names=FALSE)
        
        #Latest minister list - handy for catching changes quickly
        minister_list <- subset(roles_list, tolower(Role) %in% c('minister','first minister','deputy first minister','junior minister','speaker','deputy speaker','principal deputy speaker'))
        minister_list$AffiliationTitle[minister_list$AffiliationTitle == 'deputy First Minister'] <- 'Deputy First Minister'
        minister_list$AffiliationTitle[minister_list$AffiliationTitle == 'junior Minister'] <- 'Junior Minister in Executive Office'
        write.csv(minister_list[,c('PersonId','Organisation','AffiliationTitle')], 'data/current_ministers_and_speakers.csv', quote=TRUE, row.names=FALSE)
    }, 
    error=function(e) { message('Failed to complete update_assembly_lists; continuing') }
    )

    invisible()
}

add_pca_scores_to_tweets <- function(new_mla_tweets) {
    tweets_pca_stuff <- readRDS('./data/tweets_pca_model.RDS')
    #repeat processing from analyse_mla_tweets
    tweet_words <- new_mla_tweets %>% 
        mutate(text = sub('//t.*','', text)) %>%   #remove the //t.co at the end 
        unnest_tokens(word, text, token='tweets', to_lower=TRUE) %>% 
        select(status_id, word) %>%
        anti_join(subset(stop_words, lexicon=='SMART'), by='word')  #removes half of the 2.6m words
    #will lose some tweets here - fill in with NA later
    tweet_dtm <- tweet_words %>% 
        filter(!grepl('^@',word)) %>%
        mutate(word = sub('^#', '', word),
               word = str_replace_all(word, paste(month.name, collapse='|'), '_month_'),
               word = str_replace_all(word, paste(month.abb, collapse='|'), '_month_'),
               word = str_replace_all(word, '^\\d+$', '_number_')) %>% 
        count(status_id, word)
    #we only need the 500 tokens from the model
    tweet_dtm <- tweet_dtm %>% filter(word %in% tweets_pca_stuff[['token_list']]) %>% 
        cast_dtm(document='status_id', term='word', value='n') %>%
        as.matrix() %>% as.data.frame()
    missing_columns <- tweets_pca_stuff[['token_list']][!(tweets_pca_stuff[['token_list']] %in% names(tweet_dtm))]
    for (c in missing_columns) {
        tweet_dtm[,c] <- 0
    }
    assertthat::assert_that(all(tweets_pca_stuff[['token_list']] %in% names(tweet_dtm)))
    tweet_dtm <- tweet_dtm[, tweets_pca_stuff[['token_list']]]
    #now have a df ready to score with the pca model
    
    pca_scored_tweets <- predict(tweets_pca_stuff[['pca_model']], tweet_dtm) %>% data.frame()
    pca_scored_tweets$status_id <- row.names(pca_scored_tweets)
    pca_scored_tweets <- pca_scored_tweets[, c('status_id','PC1','PC2')]
    new_mla_tweets_w_pca <- left_join(new_mla_tweets, pca_scored_tweets, by='status_id')
    return(new_mla_tweets_w_pca)
}

get_tidy_hansardcomponents_object <- function(plenary_doc_id, session_choice) {
    #2020-2022, 2022-2027, and future will go to all_politicians_list; 
    #  hist_mla_ids contains lists up to 2020-2022 but it is all 294 members in each session so doesn't seem to have been useful
    if (session_choice < '2020-2022') {
        niassembly_mla_list <- subset(read_feather('./data/hist_mla_ids_by_session.feather'),
                                            session_name == session_choice)[,c('MemberLastName','normal_name')]
    } else {
        niassembly_mla_list <- read.csv('./data/all_politicians_list.csv')[,c('MemberLastName','normal_name')]
    }
    niassembly_mla_list <- niassembly_mla_list[!duplicated(niassembly_mla_list),]
    #Unfortunately this is not filtering to the time each member was active
    #Manually filter out some problem surnames that are inactive as of 2012
    #If there are two active with the same surname and gender, they will use an initial in Hansard,
    #  otherwise they won't, so need to filter out the inactive duplicates here.
    niassembly_mla_list <- subset(niassembly_mla_list, 
                                  !(normal_name %in% c('Fraser Agnew','Tom Hamilton','Dermot Nesbitt','Carmel Hanna',
                                                       'Ken Robinson','Mark Robinson','Iris Robinson',
                                                       'Boyd Douglas','Mark Durkan','Sam Foster',
                                                       'Paul Butler','Robert McCartney','Mick Murphy',
                                                       'Paul Maskey','Willie Clarke',  #just about left by 2012
                                                       'Billy Bell','Eileen Bell','Derek Hussey',
                                                       'Jim Wilson','Cedric Wilson','Brian Wilson',
                                                       'Gerry McHugh')))
    if (session_choice >= '2016-2017') niassembly_mla_list <- subset(niassembly_mla_list, normal_name != 'Peter Robinson')
    if (session_choice > '2016-2017') niassembly_mla_list <- subset(niassembly_mla_list, 
        !(normal_name %in% c('Eamonn McCann','Maeve McLaughlin','Mitchel McLaughlin')))
    if (session_choice < '2016-2017') niassembly_mla_list <- subset(niassembly_mla_list, normal_name != 'Keith Buchanan')
    
    tmp <- GET(sprintf('http://data.niassembly.gov.uk/hansard.asmx/GetHansardComponentsByReportId_JSON?reportId=%s',
                       plenary_doc_id))
    if (grepl('Data at the root level', tmp)) {
        cat('Bad response for doc id',plenary_doc_id,'\n')
        return(data.frame())
    }
    hcl_df <- fromJSON(content(tmp, as='text'))$AllHansardComponentsList$HansardComponent
    
    hansard_contribs <- data.frame()
    spoken_text <- ''
    next_speaker <- ''
    for (i in seq_along(hcl_df$ComponentId)) {
        #print(tmp$ComponentType[i])
        if (hcl_df$ComponentType[i] == 'Time') next()
        
        if (grepl('Speaker', hcl_df$ComponentType[i])) {
            if (spoken_text != '' & next_speaker != 'skip') {
                #print('saving')
                next_speaker <- gsub('\\(.*\\)','',next_speaker)
                speaker_split <- strsplit(gsub(':','',next_speaker),' ')[[1]]
                this_last_name <- speaker_split[length(speaker_split)]
                
                if (this_last_name == "Chuilín") this_last_name <- "Ní Chuilín"
                if (this_last_name == "Muilleoir") this_last_name <- "Ó Muilleoir"
                if (this_last_name == "Pengelly") this_last_name <- "Little Pengelly"
                if (this_last_name == "Morrow") this_last_name <- "Morrow of Clogher Valley"
                if (this_last_name == "hOisín") this_last_name <- "Ó hOisín"
                if (this_last_name == "Empey") this_last_name <- "Empey of Shandon"
                #if (this_last_name == "hOisín") this_last_name <- "Ó hOisín"
                if (paste(speaker_split, collapse=' ') == 'Mrs Cameron') {
                    if (!('Cameron') %in% niassembly_mla_list$MemberLastName) this_last_name <- 'Brown'
                } 
                
                speaker_convert <- subset(niassembly_mla_list, MemberLastName == this_last_name)$normal_name

                if (length(speaker_convert) > 1) {
                    #cat(speaker_convert,'\n')
                    #cat(speaker_split,'\n')
                    initials <- as.character(sapply(speaker_convert, function(n) substr(trimws(strsplit(n, ' ')[[1]][1]),1,1)))
                    #could be wrong if there are two with same initial
                    speaker_convert <- speaker_convert[which(initials==speaker_split[2])[1]]
                    
                    if (is.na(speaker_convert)) {
                        speaker_split <- paste(speaker_split, collapse=' ')
                        #cat(speaker_split,(speaker_split=="Mrs O'Neill"), str(speaker_split), '\n')
                        #Try to correct some where the initial is not used but gender gives it away
                        if (speaker_split == "Ms Gildernew") speaker_convert <- 'Michelle Gildernew'
                        if (speaker_split == "Mr Gildernew") speaker_convert <- 'Colm Gildernew'
                        if (speaker_split == "Ms Mallon") speaker_convert <- 'Nichola Mallon'
                        if (speaker_split == "Ms Armstrong") speaker_convert <- 'Kellie Armstrong'
                        if (speaker_split == "Mrs O'Neill") speaker_convert <- "Michelle O'Neill"
                        if (speaker_split == "Ms Gildernew") speaker_convert <- 'Michelle Gildernew'
                        if (speaker_split == "Ms Anderson") speaker_convert <- 'Martina Anderson'
                        if (speaker_split == "Mr Anderson") speaker_convert <- 'Sydney Anderson'
                        if (speaker_split == "Mr Mullan") speaker_convert <- 'Gerry Mullan'
                        if (speaker_split == "Ms Mullan") speaker_convert <- 'Karen Mullan'
                        if (speaker_split == "Mr McIlveen") speaker_convert <- 'David McIlveen'
                        if (speaker_split == "Miss McIlveen") speaker_convert <- 'Michelle McIlveen'
                        if (speaker_split == "Mr Kelly") speaker_convert <- 'Gerry Kelly'
                        if (speaker_split == 'Mr Ramsey') speaker_convert <- 'Pat Ramsey'
                        if (speaker_split == 'Ms Ramsey') speaker_convert <- 'Sue Ramsey'
                        if (speaker_split == 'Ms Maeve McLaughlin') speaker_convert <- 'Maeve McLaughlin'
                        if (speaker_split == 'Mr McLaughlin') speaker_convert <- 'Mitchel McLaughlin'
                        if (speaker_split == 'Ms Ennis') speaker_convert <- 'Sinéad Ennis'
                        if (speaker_split == 'Mr Ennis') speaker_convert <- 'George Ennis'
                        if (speaker_split == 'Ms McCann') speaker_convert <- 'Jennifer McCann'
                        
                        #if (speaker_split == 'Mrs Cameron') speaker_convert <- 'Pam Cameron'
                        #cat(speaker_convert,'\n')
                        #Can't be sure of Robinson sometimes, 
                    }
                }
                
                #cat(length(speaker_convert), speaker_convert, speaker_split, this_last_name,'\n')
                if (length(speaker_convert) == 1 && !is.na(speaker_convert)) {
                    hansard_contribs <- rbind(hansard_contribs, data.frame(speaker = speaker_convert, 
                                                                           contrib = trimws(spoken_text)))    
                } else {
                    cat('Didn\'t recognise surname',this_last_name,speaker_split,'\n')
                    hansard_contribs <- rbind(hansard_contribs, data.frame(speaker = paste('unknown',this_last_name,sep='-'),
                                                                           contrib = trimws(spoken_text)))
                }
                
            }
            
            next_speaker <- (if (hcl_df$ComponentType[i] %in% c('Speaker (MlaName)','Speaker (MinisterAndName)')) hcl_df$ComponentText[i]
                             else 'skip')
            #print('resetting')
            spoken_text <- ''
        } else if (hcl_df$ComponentType[i] %in% c('Spoken Text','Quote','Plenary Item Text') & next_speaker!='') {
            #print('appending')
            spoken_text <- paste(spoken_text, gsub('\r<BR />\r<BR />',' ',hcl_df$ComponentText[i]))
        }
        #cat(hcl_df$ComponentType[i],next_speaker,'\n')
    }
    if (nrow(hansard_contribs) > 0) {
        hansard_contribs$session_id <- plenary_doc_id
        hansard_contribs$seq_id <- seq_along(hansard_contribs$contrib)
    }
    hansard_contribs
}
