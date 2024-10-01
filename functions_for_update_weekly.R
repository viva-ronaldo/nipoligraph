#Functions to update various parts of the data

library(httr)
library(jsonlite)
#library(feather)
library(arrow)  # for read_feather to work on file written from python
library(xml2)
library(lubridate)

# Check for new MLAs on AIMs
update_and_get_politicians_list <- function(politicians_list_filepath) {
    politicians <- read.csv(politicians_list_filepath)
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
        write.csv(politicians, politicians_list_filepath, row.names=FALSE)
        
        #Make a note of this change
        write.table(data.frame(d=as.character(Sys.Date()), t=sprintf('Added new MLA(s) %s', paste(new_mlas$MemberLastName, collapse=','))), 
                    'politician_list_changes.log', 
                    append=TRUE, quote=FALSE, sep=' ', col.names=FALSE, row.names=FALSE)
        
    }
    #Also check for any MLAs that have left and set them to inactive in the file
    newly_inactive_mlas <- subset(politicians, role=='MLA' & active==1 & !(MemberName %in% mlas$MemberName))
    if (nrow(newly_inactive_mlas) > 0) {
        politicians %<>% mutate(active = ifelse(MemberName %in% newly_inactive_mlas$MemberName, 0, active))
        write.csv(politicians, politicians_list_filepath, row.names=FALSE)
        write.table(data.frame(d=as.character(Sys.Date()), t=sprintf('Removed inactive MLA(s) %s', paste(newly_inactive_mlas$MemberLastName, collapse=','))), 
                    'politician_list_changes.log', 
                    append=TRUE, quote=FALSE, sep=' ', col.names=FALSE, row.names=FALSE)
    }
    return(politicians)
}

# Get the latest Assembly event diary, committee and minister lists
update_assembly_lists <- function(diary_filepath,
                                  committees_list_filepath,
                                  minister_list_filepath,
                                  mla_interests_list_filepath) {
    tryCatch({
        #Diary of business
        tmp <- GET(sprintf('http://data.niassembly.gov.uk/plenary.asmx/GetBusinessDiary_JSON?startDate=%s&endDate=%s',
                           Sys.Date(), Sys.Date()+365))
        diary <- fromJSON(content(tmp, as='text'))$BusinessDiary$DiaryItem
        write.table(diary, diary_filepath, quote=FALSE, row.names=FALSE, sep='|')
        
        #Latest committee memberships 
        tmp <- GET('http://data.niassembly.gov.uk/members.asmx/GetAllMemberRoles_JSON?')
        roles_list <- fromJSON(content(tmp, as='text'))$AllMembersRoles$Role
        committees_list <- subset(roles_list,
                                  Role %in% c('Committee Member', 'Committee Chair', 'Committee Deputy Chair'))
        write.csv(committees_list[, c('PersonId', 'Organisation', 'Role')],
                  committees_list_filepath, quote=TRUE, row.names=FALSE)
        
        #Latest minister list - handy for catching changes quickly
        minister_list <- subset(roles_list,
                                tolower(Role) %in% c('minister', 'first minister',
                                                     'deputy first minister', 'junior minister',
                                                     'speaker', 'deputy speaker', 'principal deputy speaker'))
        minister_list$AffiliationTitle[
            minister_list$AffiliationTitle == 'deputy First Minister'] <- 'Deputy First Minister'
        minister_list$AffiliationTitle[
            minister_list$AffiliationTitle == 'junior Minister'] <- 'Junior Minister in Executive Office'
        write.csv(minister_list[, c('PersonId', 'Organisation', 'AffiliationTitle')],
                  minister_list_filepath, quote=TRUE, row.names=FALSE)
        
        #Register of interests (JSON API currently broken)
        tmp <- GET(sprintf('http://data.niassembly.gov.uk/register.asmx/GetAllRegisteredInterests'))
        mla_interests_xml <- read_xml(content(tmp, as='text'))
        mla_interests_df <- sapply(1:length(xml_children(mla_interests_xml)),
                                   function(i) unlist(as_list(xml_children(mla_interests_xml)[i]))) %>%
            t() %>% as.data.frame() %>%
            filter(!(RegisterEntry %in% c('None', 'None.', 'none', ' ', '')))
        write.csv(mla_interests_df, mla_interests_list_filepath, quote=TRUE, row.names=FALSE)
    }, 
    error=function(e) { message('Failed to complete update_assembly_lists; continuing') }
    )
    invisible()
}

# Top-level function for updating questions and answers
update_questions_and_answers <- function(questions_filepath, answers_filepath) {
    # Get latest questions from AIMS
    tmp <- GET(sprintf('http://data.niassembly.gov.uk/questions.asmx/GetQuestionsForOralAnswer_TabledInRange_JSON?startDate=%s&endDate=%s',
                       Sys.Date()-21, Sys.Date()+500))
    questions_oral <- fromJSON(content(tmp, as='text'))$QuestionsList$Question
    tmp2 <- GET(sprintf('http://data.niassembly.gov.uk/questions.asmx/GetQuestionsForWrittenAnswer_TabledInRange_JSON?startDate=%s&endDate=%s',
                        Sys.Date()-21, Sys.Date()+500))
    questions_written <- fromJSON(content(tmp2, as='text'))$QuestionsList$Question
    
    # Append new ones to the question file
    update_questions_file(questions_oral, questions_written, questions_filepath)
    
    # Append any new answers to pending questions to the answers file
    update_answers_file(questions_filepath, answers_filepath)
}

# Update questions file after retrieving recent questions
update_questions_file <- function(questions_oral, questions_written, questions_filepath) {
    new_questions <- data.frame()
    #Some questions can be asked to Assembly Commission, which has no MinisterPersonId,
    #  and if these are the only questions found, MinisterPersonId is omitted from the response.
    #  We can skip those questions (is.na(MinisterPersonId)) but skip entirely if MinisterPersonId is not in response.
    if (!is.null(questions_oral) & 'MinisterPersonId' %in% names(questions_oral)) {
        questions_oral$RequestedAnswerType <- 'oral'
        new_questions <- rbind(new_questions,
                               questions_oral %>% 
                                   filter(!is.na(MinisterPersonId)) %>%
                                   select(DocumentId, TabledDate, TablerPersonId, MinisterPersonId,
                                          QuestionText, RequestedAnswerType))
    }
    if (!is.null(questions_written) & 'MinisterPersonId' %in% names(questions_written)) {
        questions_written$RequestedAnswerType <- 'written'
        new_questions <- rbind(new_questions,
                               questions_written %>% 
                                   filter(!is.na(MinisterPersonId)) %>%
                                   select(DocumentId, TabledDate, TablerPersonId, MinisterPersonId,
                                          QuestionText, RequestedAnswerType))
    }
    
    existing_questions <- read_feather(questions_filepath)
    if (nrow(new_questions) > 0) {
        existing_questions <- rbind(existing_questions, new_questions) %>%
            filter(!duplicated(.[, c('DocumentId', 'MinisterPersonId')]))
        message('- Writing Assembly questions')
        write_feather(existing_questions, questions_filepath)
    } else {
        message('- No new Assembly questions found')
    }
}

# Look for new answers to questions in the questions file that aren't in the answers file,
#   and update the answers file with any new answers found.
update_answers_file <- function(questions_filepath, answers_filepath) {
    existing_questions <- read_feather(questions_filepath)
    existing_answers <- read_feather(answers_filepath)
    pending_questions <- existing_questions %>%
        filter(!DocumentId %in% existing_answers$DocumentId) %>% pull(DocumentId)
    # Some questions were answered after a gap of almost 2 years so can't really
    #   skip the 2022 ones, have to check them every time. Now not a problem as using fast method.
    
    # Get all questions answered in the last 3 months and filter to those pending here;
    #   these APIs contain all the fields needed for the answers table.
    tmp <- GET(sprintf('http://data.niassembly.gov.uk/questions.asmx/GetQuestionsForWrittenAnswer_AnsweredInRange_JSON?startDate=%s&endDate=%s',
                       today()-90, today()+1))
    tmp <- fromJSON(content(tmp, as='text'))$QuestionsList$Question
    new_written_answers <- tmp %>%
        filter(DocumentId %in% pending_questions) %>% 
        select(DocumentId, DocumentType, TablerName, TablerPersonId,
               TabledDate, AnsweredOnDate, PriorityRequest,
               MinisterPersonId, MinisterTitle)
    tmp <- GET(sprintf('http://data.niassembly.gov.uk/questions.asmx/GetQuestionsForOralAnswer_AnsweredInRange_JSON?startDate=%s&endDate=%s',
                       today()-90, today()+1))
    tmp <- fromJSON(content(tmp, as='text'))$QuestionsList$Question
    new_oral_answers <- tmp %>%
        filter(DocumentId %in% pending_questions) %>% 
        mutate(PriorityRequest='false') %>% 
        select(DocumentId, DocumentType, TablerName, TablerPersonId,
               TabledDate, AnsweredOnDate, PriorityRequest,
               MinisterPersonId, MinisterTitle)
    new_answers <- rbind(new_written_answers, new_oral_answers)
    
    if (nrow(new_answers) > 0) {
        existing_answers <- rbind(existing_answers, new_answers)
        message('- Writing Assembly answers')
        write_feather(existing_answers, answers_filepath)
    } else {
        message('- No new Assembly answers found')
    }
}

# Called by update_answers_file; search AIMS for answers to a list of pending questions
# get_answers_to_questions <- function(documentId_list) {
#     answers_to_questions <- data.frame()
#     for (q_id in unique(documentId_list)) {
#         tmp <- GET(sprintf('http://data.niassembly.gov.uk/questions.asmx/GetQuestionDetails_JSON?documentId=%s', q_id))
#         tmp <- fromJSON(content(tmp, as='text'))$QuestionsList$Question
#         
#         if (is.null(tmp)) next
# 
#         if (!is.data.frame(tmp)) {
#             #occasionally need to handle a NULL here
#             #tmp <- lapply(tmp, function(x) ifelse(is.null(x), NA, x))
#             tmp <- data.frame(tmp)
#         }
#         
#         #Can get duplicates- looks like where one answer should be appended to another
#         #OR can be minister and deputy first minister getting a row each as MinisterPersonId - don't concat these
#         if (nrow(tmp) > 1 && 
#             (('MinisterPersonId' %in% names(tmp) && n_distinct(tmp$MinisterPersonId) == 1) ||
#              (tmp$MinisterTitle == 'Assembly Commission'))) {
#             full_answer <- paste(tmp$AnswerPlainText, sep=' _PLUS_ ')
#             tmp <- tmp[order(tmp$AnsweredOnDate), ]
#             tmp <- tmp[1, ]
#             cat(sprintf('concatenated answers for document id %s\n', tmp$DocumentId[1]))
#         }
#         
#         #Not all have a MinisterPersonId - they are Ministertitle == Assembly Commission
#         if (!('MinisterPersonId' %in% names(tmp))) tmp$MinisterPersonId <- NA
#         
#         #If AnswerByDate is present instead of AnsweredOnDate, it hasn't been answered yet
#         if ('AnsweredOnDate' %in% names(tmp)) {
#             answers_to_questions <- rbind(answers_to_questions, 
#                                           tmp %>% select(DocumentId, DocumentType, TablerName, TablerPersonId,
#                                                          TabledDate, AnsweredOnDate, PriorityRequest,
#                                                          MinisterPersonId, MinisterTitle))
#         }
#     }
#     return(answers_to_questions)
# }

# Division votes - just check what the most recent date was and go from there
update_vote_list <- function(vote_details_filepath, vote_results_filepath) {
    existing_votes <- read_feather(vote_details_filepath)
    existing_vote_results <- read_feather(vote_results_filepath)
    message(sprintf('- Latest existing vote date = %s', max(existing_votes$DivisionDate)))
    
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
        
        cat(sprintf('Found %i possible new votes\n', nrow(possible_new_votes)))
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
                new_row <- data.frame(division_result[c('EventId', 'Title', 'DecisionMethod',
                                                        'DecisionType', 'Outcome')])
                new_row['tabler_personIDs'] <- paste(tabler_personIDs$TablerPersonID, collapse=';')
                new_row['true_DivisionDate'] <- vote_date
                new_vote_summary <- rbind(new_vote_summary, new_row)
                
                # Get the member votes
                tmp <- GET(sprintf('http://data.niassembly.gov.uk/plenary.asmx/GetDivisionMemberVoting_JSON?documentId=%s',d_id))
                member_votes <- fromJSON(content(tmp, as='text'))$MemberVoting$Member
                new_vote_results <- rbind(new_vote_results,
                                          member_votes[, c('EventID', 'PersonID', 'Vote', 'Designation')])
            }
            names(new_vote_results)[names(new_vote_results)=='EventID'] <- 'EventId'
            names(new_vote_results)[names(new_vote_results)=='PersonID'] <- 'PersonId'
            
            #replace the dodgy DivisionDate with the better PlenaryDate=true_DivisionDate
            new_vote_summary <- left_join(possible_new_votes, new_vote_summary, by='EventId') %>%
                mutate(DivisionDate=true_DivisionDate) %>% select(-true_DivisionDate)
            
            existing_votes <- rbind(existing_votes, new_vote_summary)
            existing_vote_results <- rbind(existing_vote_results, new_vote_results)
            write_feather(existing_votes, vote_details_filepath)
            write_feather(existing_vote_results, vote_results_filepath)
        }
    }
    invisible()
}

# Plenary contributions
update_plenary_contribs <- function(contribs_filepath,
                                    current_session_name,
                                    politicians_list_filepath,
                                    hist_mlas_by_session_filepath) {
    existing_contribs <- read_feather(contribs_filepath)
    
    tryCatch({
        tmp <- GET('http://data.niassembly.gov.uk/hansard.asmx/GetAllHansardReports_JSON?')
        new_reports_list <- fromJSON(content(tmp, as='text'))$AllHansardComponentsList$HansardComponent %>%
            filter(PlenarySessionName >= current_min_session_name_for_plenary, !(ReportDocId %in% existing_contribs$ReportDocId))
        
        new_contribs <- data.frame()
        for (doc_id in new_reports_list$ReportDocId) {
            session_hansard_contribs <- get_tidy_hansardcomponents_object(doc_id,
                                                                          current_session_name,
                                                                          politicians_list_filepath,
                                                                          hist_mlas_by_session_filepath)
            #at least one plenary was just two minutes' silence so returns 0 rows
            if (nrow(session_hansard_contribs) > 0) {
                session_hansard_contribs$ReportDocId <- doc_id
                session_hansard_contribs <- left_join(session_hansard_contribs,
                                                      new_reports_list,
                                                      by='ReportDocId')
            }
            new_contribs <- rbind(new_contribs, session_hansard_contribs)
        }
        message('- Writing assembly contribs')
        write_feather(rbind(existing_contribs, new_contribs), contribs_filepath)
        
        # Manually try to avoid unmatched speakers
        #contribs %>% filter(grepl('unknown', speaker)) %>% pull(speaker) %>% unique()
    },
    error=function(e) { message('Failed to complete update_plenary_contribs; continuing') }
    )
    invisible()
}

# Parse the Hansard documents into contributions by normal_name speaker
get_tidy_hansardcomponents_object <- function(plenary_doc_id,
                                              session_choice,
                                              politicians_list_filepath,
                                              hist_mlas_by_session_filepath) {
    #2020-2022, 2022-2027, and future will go to all_politicians_list; 
    #  hist_mla_ids contains lists up to 2020-2022 but it is all 294 members in each session so doesn't seem to have been useful
    if (session_choice < '2020-2022') {
        niassembly_mla_list <- subset(read_feather(hist_mlas_by_session_filepath),
                                      session_name == session_choice)
    } else {
        niassembly_mla_list <- read.csv(politicians_list_filepath)
    }
    niassembly_mla_list <- subset(niassembly_mla_list, role=='MLA')
    niassembly_mla_list <- niassembly_mla_list[!duplicated(niassembly_mla_list), c('MemberLastName', 'normal_name', 'active')]
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
    # For current session, we can use the active filter
    if (session_choice >= '2022') niassembly_mla_list <- subset(niassembly_mla_list, active==1)
    
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
        if (hcl_df$ComponentType[i] == 'Time') next()
        
        if (grepl('Speaker', hcl_df$ComponentType[i])) {
            if (spoken_text != '' & next_speaker != 'skip') {
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
                speaker_convert <- unique(speaker_convert)

                # This matches surname only so there can be more than one match
                if (length(speaker_convert) > 1) {
                    initials <- as.character(sapply(speaker_convert, function(n) substr(trimws(strsplit(n, ' ')[[1]][1]),1,1)))
                    #could be wrong if there are two with same initial
                    speaker_convert <- speaker_convert[which(initials==speaker_split[2])[1]]
                    
                    if (is.na(speaker_convert)) {
                        speaker_split <- paste(speaker_split, collapse=' ')
                        #Try to correct some where the initial is not used but gender gives it away
                        if (speaker_split == "Ms Gildernew") speaker_convert <- 'Michelle Gildernew'
                        if (speaker_split == "Mr Gildernew") speaker_convert <- 'Colm Gildernew'
                        if (speaker_split == "Ms Mallon") speaker_convert <- 'Nichola Mallon'
                        if (speaker_split == "Ms Armstrong") speaker_convert <- 'Kellie Armstrong'
                        if (speaker_split == "Mrs O'Neill") speaker_convert <- "Michelle O'Neill"
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
                        if (speaker_split == 'Ms McLaughlin') speaker_convert <- 'Sinéad McLaughlin'
                        if (speaker_split == 'Mr Dunne') speaker_convert <- 'Stephen Dunne'
                        if (speaker_split == 'Mr Bradley') speaker_convert <- 'Maurice Bradley'
                    }
                }
                
                if (length(speaker_convert) == 1 && !is.na(speaker_convert)) {
                    hansard_contribs <- rbind(hansard_contribs, data.frame(speaker = speaker_convert, 
                                                                           contrib = trimws(spoken_text)))    
                } else {
                    cat(sprintf('Didn\'t recognise surname %s from speaker_split=%s\n', this_last_name, speaker_split))
                    hansard_contribs <- rbind(hansard_contribs, data.frame(speaker = paste('unknown', this_last_name, sep='-'),
                                                                           contrib = trimws(spoken_text)))
                }
            }
            
            next_speaker <- (if (hcl_df$ComponentType[i] %in% c('Speaker (MlaName)','Speaker (MinisterAndName)')) hcl_df$ComponentText[i]
                             else 'skip')
            spoken_text <- ''   # reset for next iteration
        } else if (hcl_df$ComponentType[i] %in% c('Spoken Text','Quote','Plenary Item Text') & next_speaker!='') {
            spoken_text <- paste(spoken_text, gsub('\r<BR />\r<BR />', ' ', hcl_df$ComponentText[i]))
        }
    }
    if (nrow(hansard_contribs) > 0) {
        hansard_contribs$session_id <- plenary_doc_id
        hansard_contribs$seq_id <- seq_along(hansard_contribs$contrib)
    }
    hansard_contribs
}

update_average_contrib_emotions <- function(contribs_filepath, contrib_emotions_filepath) {
    # Uses current contribs file - previous session should be moved to a hist file every few years
    contribs <- read_feather(contribs_filepath)
    cat(sprintf('Contrib emotions are averaged over %i plenary sessions from %s to %s\n',
                n_distinct(contribs$session_id),
                substr(min(contribs$PlenaryDate), 1, 10),
                substr(max(contribs$PlenaryDate), 1, 10)))
    
    speakers <- unique(contribs$speaker)
    
    #Do in chunks to work with limited memory; output is 16 rows per speaker
    mla_emotions <- contribs %>% filter(speaker %in% speakers[1:min(length(speakers), 10)]) %>% 
        mutate(split_text = get_sentences(contrib)) %$%
        emotion_by(split_text, by=speaker)  # uses sentimentr::emotion_by
    if (length(speakers) > 10) {
        for (i in seq(11, length(speakers), 10)) {
            mla_emotions <- rbind(mla_emotions,
                                  contribs %>% filter(speaker %in% speakers[i:min(length(speakers), i+10)]) %>% 
                                      mutate(split_text = get_sentences(contrib)) %$%
                                      emotion_by(split_text, by=speaker))
        }
    }
    write_feather(mla_emotions, contrib_emotions_filepath)
}

# Get Irish News and News Letter stories from Twitter - now skipped in 2023
update_news_from_twitter <- function(politicians, existing_news_articles_filepath) {
    
    existing_news_articles <- read_feather(existing_news_articles_filepath)
    
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
}


# Update the news_articles_w_sentiment file with any new entries in news_articles file
update_news_article_sentiment <- function(news_articles_filepath, news_articles_w_sentiment_filepath) {
    news_articles <- read_feather(news_articles_filepath)
    news_articles_with_sentiment <- read_feather(news_articles_w_sentiment_filepath)
    new_news_articles <- anti_join(news_articles, news_articles_with_sentiment, 
                                   by=c('normal_name', 'published_date', 'title'))
    message(sprintf('- %i new articles to score for sentiment', nrow(new_news_articles)))
    
    if (nrow(new_news_articles) > 0) {
        new_news_articles$article_id <- seq_along(new_news_articles$normal_name)
        # Makes only one row per article so memory shouldn't be a problem; can do in chunks if needed.
        new_news_sentiments <- new_news_articles %>%
            mutate(summary_split = get_sentences(summary)) %$%
            sentiment_by(summary_split, list(normal_name,article_id),
                         polarity_dt = lexicon::hash_sentiment_jockers_rinker %>% filter(!grepl('^econo|justice|money|assembly|traditional|socialist|progressive|conservative|voter|elect|holiday|guardian|star|independence|united',x)))
        new_news_articles <- left_join(new_news_articles,
                                       new_news_sentiments %>% select(article_id, sr_sentiment_score=ave_sentiment),
                                       by='article_id') %>%
            select(-article_id)
        
        news_articles_with_sentiment <- rbind(news_articles_with_sentiment, 
                                              new_news_articles[, c('normal_name', 'published_date', 'source',
                                                                    'link', 'title', 'sr_sentiment_score')])
        message('- Writing news slim sentiment')
        write_feather(news_articles_with_sentiment,
                      news_articles_w_sentiment_filepath)
    }
}




# Moved to python, now skipped anyway
# add_pca_scores_to_tweets <- function(new_mla_tweets) {
#     tweets_pca_stuff <- readRDS('./data/tweets_pca_model.RDS')
#     #repeat processing from analyse_mla_tweets
#     tweet_words <- new_mla_tweets %>% 
#         mutate(text = sub('//t.*','', text)) %>%   #remove the //t.co at the end 
#         unnest_tokens(word, text, token='tweets', to_lower=TRUE) %>% 
#         select(status_id, word) %>%
#         anti_join(subset(stop_words, lexicon=='SMART'), by='word')  #removes half of the 2.6m words
#     #will lose some tweets here - fill in with NA later
#     tweet_dtm <- tweet_words %>% 
#         filter(!grepl('^@',word)) %>%
#         mutate(word = sub('^#', '', word),
#                word = str_replace_all(word, paste(month.name, collapse='|'), '_month_'),
#                word = str_replace_all(word, paste(month.abb, collapse='|'), '_month_'),
#                word = str_replace_all(word, '^\\d+$', '_number_')) %>% 
#         count(status_id, word)
#     #we only need the 500 tokens from the model
#     tweet_dtm <- tweet_dtm %>% filter(word %in% tweets_pca_stuff[['token_list']]) %>% 
#         cast_dtm(document='status_id', term='word', value='n') %>%
#         as.matrix() %>% as.data.frame()
#     missing_columns <- tweets_pca_stuff[['token_list']][!(tweets_pca_stuff[['token_list']] %in% names(tweet_dtm))]
#     for (c in missing_columns) {
#         tweet_dtm[,c] <- 0
#     }
#     assertthat::assert_that(all(tweets_pca_stuff[['token_list']] %in% names(tweet_dtm)))
#     tweet_dtm <- tweet_dtm[, tweets_pca_stuff[['token_list']]]
#     #now have a df ready to score with the pca model
#     
#     pca_scored_tweets <- predict(tweets_pca_stuff[['pca_model']], tweet_dtm) %>% data.frame()
#     pca_scored_tweets$status_id <- row.names(pca_scored_tweets)
#     pca_scored_tweets <- pca_scored_tweets[, c('status_id','PC1','PC2')]
#     new_mla_tweets_w_pca <- left_join(new_mla_tweets, pca_scored_tweets, by='status_id')
#     return(new_mla_tweets_w_pca)
# }


make_twitter_network_files <- function(twitter_ids,
                                       network_nodes_filepath,
                                       network_edges_filepath,
                                       network_top5s_filepath) {
    
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
    
    all_mla_tweets <- read_feather('TODO_slim_tweets_filepath')
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
    
    #get x and y on right scale for Vega
    tmp <- myplot$data
    tmp$x <- tmp$x * 650
    tmp$y <- tmp$y * 500
    message('Writing tweets network files - nodes .json, edges .csv, and top 5s .csv')
    write_json(tmp, network_nodes_filepath)
    tmp <- edges
    names(tmp) <- c('source', 'target', 'value')
    #write.csv(tmp, 'flask/static/tweets_network_since1july2020_edges.csv', quote=FALSE, row.names=FALSE)
    write.csv(tmp, network_edges_filepath, quote=FALSE, row.names=FALSE)
    
    #Get top 5s of network measures
    myplot$data$betw_centr <- sna::betweenness(retweets_network, cmode='directed')
    myplot$data$info_centr <- sna::infocent(retweets_network) 
    myplot$data$page_rank <- page_rank(asIgraph(retweets_network))$vector
    data.frame(rank=seq(5), 
               info_centr = arrange(myplot$data, -info_centr) %>% head(5) %>% pull(plot_label_all),
               page_rank = arrange(myplot$data, -page_rank) %>% head(5) %>% pull(plot_label_all),
               betw_centr = arrange(myplot$data, -betw_centr) %>% head(5) %>% pull(plot_label_all)) %>%
        write.csv(network_top5s_filepath, quote=FALSE, row.names=FALSE)
    
    #rm(all_mla_tweets, new_mla_tweets, existing_mla_tweets, all_mla_tweets_for_plot, tmp)
    invisible()
}
