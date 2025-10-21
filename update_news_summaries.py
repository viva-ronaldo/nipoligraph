import getpass, json, os, time, yaml
import pandas as pd
import openai
from openai import OpenAI
from datetime import datetime, timedelta
import boto3  # if data_dir is S3

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

time_limit_seconds = 60 * 9

start_time = time.time()

if getpass.getuser() == 'david':
    oaikey_path = os.path.expanduser('~/oaikey.txt')
else:
    oaikey_path = './oaikey.txt'
with open(oaikey_path, 'r') as f:
    openai.api_key = f.read().strip()
client = OpenAI(api_key=openai.api_key)

data_dir = f"s3://{config['NIPOL_DATA_BUCKET']}/"

#news_path = os.path.join(data_dir, 'newscatcher_articles_sep2020topresent.feather')
news_path = os.path.join(data_dir, 'worldnewsapi_articles_oct2025topresent.feather')
news_summaries_path = os.path.join(data_dir, 'news_summaries.feather')

refresh_frequency_days = config['NEWS_SUMMARIES_REFRESH_FREQUENCY_DAYS']
model = config['NEWS_SUMMARIES_OAI_MODEL']
input_token_cost_usd, output_token_cost_usd = 0.15/1e6, 0.60/1e6

news = pd.read_feather(news_path)
news['published_date_short'] = news.published_date.apply(lambda s: s[:10])

if 's3' in data_dir:
    s3 = boto3.client('s3')
    bucket_name = data_dir.replace('s3://', '').split('/')[0]
    response = s3.get_object(Bucket=bucket_name, Key='news_source_pprint_dict.json')
    news_source_pprint_dict = json.loads(response['Body'].read().decode('utf-8'))
else:
    with open(os.path.join(data_dir, 'news_source_pprint_dict.json'), 'r') as f:
        news_source_pprint_dict = json.load(f)
news['source_pprint'] = news.source.apply(lambda s: news_source_pprint_dict.get(s, s))

news_summaries = pd.read_feather(news_summaries_path)

active_politicians = (pd.read_csv(os.path.join(data_dir, 'all_politicians_list.csv'))
                      .query('active == 1')
                      [['normal_name', 'PartyName']])
# Filter news to active politicians, and add PartyName
news = news.merge(active_politicians, on='normal_name', how='inner')

def get_news_summary_llm_prompt(
    news_articles_df,
    article_format,
    max_n_tokens=50000,
    single_article_max_tokens=3000):
    '''
    Generate a prompt for an LLM summarising the news coverage of a politician
    Arguments:
        news_articles_df: pd.DataFrame, the news articles
        article_format: 'title' or 'summary'
        max_n_tokens: int, maximum number of tokens to use in the prompt;
          gpt-3.5-turbo-16k can handle 16,000; newer models 128,000
    '''
    politician_names = news_articles_df.normal_name.unique()
    assert len(politician_names) == 1
    politician_name = politician_names[0]
    party_name = news_articles_df.PartyName.iloc[0]
    party_membership_str = 'an Independent politician, not affiliated to any party' if party_name == 'Independent' else \
        (f'a member of the {party_name}' if party_name[-5:] == 'Party' else f'a member of {party_name}')

    if article_format == 'title':
        this_formatted_articles = '\n'.join(news_articles_df.apply(lambda row: f"- {row['title']}", axis=1).unique().tolist())
        this_prompt = f"The Northern Ireland politician {politician_name}, who is {party_membership_str}, has recently been mentioned in news articles "\
            f"with titles listed below. Summarise, in one paragraph, the activity of {politician_name} during this period. "\
            f"Remember that although {politician_name} is mentioned in the text of each of the articles (not shown here), some of the titles may not refer to {politician_name}. "\
            f"Start the response with \'{politician_name} has\'.\n---\n{this_formatted_articles}.\n"
        
    elif article_format == 'summary':
        # Skip any with summary over ~3000 tokens (these use up too much of the token budget; this is ~2% of all summaries)
        usable_articles = news_articles_df[news_articles_df.summary.apply(lambda s: len(s)/4.5 < single_article_max_tokens)]
        # Sort by date and keep no more than the most recent ~max_n_tokens tokens
        usable_articles = (usable_articles
                           .sort_values('published_date', ascending=False)
                           .assign(approx_n_tokens = lambda df: df.summary.str.len()/4.5)
                           .assign(cumsum_tokens = lambda df: df.approx_n_tokens.cumsum())
                           .sort_values('published_date', ascending=True)
                           .query(f'cumsum_tokens < {max_n_tokens*0.9}')
                           )

        this_formatted_articles = '\n'.join(usable_articles.apply(lambda row: f"- TITLE: {row.title}\n  SOURCE: {row.source_pprint}\n  DATE: {row.published_date_short}\n  CONTENT: {row.summary}", axis=1).tolist())
        if len(usable_articles) > 2:
            this_prompt = f"The Northern Ireland politician {politician_name}, who is {party_membership_str}, has recently been mentioned in {len(usable_articles)} news articles "\
                f"with the title, source (newspaper, website, etc.), date, and text listed below. "\
                f"\n\n---\n{this_formatted_articles}.\n---\n\n"\
                f"Summarise, in 1-4 paragraphs, the activity of {politician_name} during this period. "\
                "Quote the source(s) for the stories where appropriate. "\
                f"Start the response with \'{politician_name} has\'."
        else:
            this_prompt = f"The Northern Ireland politician {politician_name}, who is {party_membership_str}, has recently been mentioned in {len(usable_articles)} news articles "\
                f"with the title, source (newspaper, website, etc.), date, and text listed below. "\
                f"\n\n---\n{this_formatted_articles}.\n---\n\n"\
                f"Summarise, in a few sentences, the activity of {politician_name} during this period. "\
                f"Start the response with \'{politician_name} has\'."

    return this_prompt

current_date = datetime.now()
one_month_back_date = current_date - timedelta(days=30)
two_months_back_date = current_date - timedelta(days=60)
three_months_back_date = current_date - timedelta(days=90)
current_date_str = current_date.strftime('%Y-%m-%d')
one_month_back_date_str = one_month_back_date.strftime('%Y-%m-%d')
two_months_back_date_str = two_months_back_date.strftime('%Y-%m-%d')
three_months_back_date_str = three_months_back_date.strftime('%Y-%m-%d')

# Get politicians with no summary in last 2 weeks
latest_news_summary_dates = news_summaries.groupby('normal_name').create_date.max().reset_index()
nn_requiring_update = (news.drop_duplicates(subset='normal_name')[['normal_name']]
                       .merge(latest_news_summary_dates, how='left', on='normal_name')
                       .assign(time_since_summary = lambda df: (current_date - df.create_date).dt.days.fillna(999))
                       .query(f'time_since_summary > {refresh_frequency_days}')
                       .normal_name.tolist()
                      )
#nn_requiring_update = [p for p in nn_requiring_update if p in active_politicians]
print(f'{len(nn_requiring_update)} politicians with news articles and no summary in last {refresh_frequency_days} days')

res = []
for i, nn in enumerate(nn_requiring_update, start=1):
    ss_last1m = news[(news.published_date_short >= one_month_back_date_str) &
        (news.normal_name == nn)]
    ss_last2m = news[(news.published_date_short >= two_months_back_date_str) &
        (news.normal_name == nn)]
    ss_last3m = news[(news.published_date_short >= three_months_back_date_str) &
        (news.normal_name == nn)]
    print(f'{i:3g}: {nn} {len(ss_last1m)} articles in last month; {len(ss_last2m)} in last 2 months; {len(ss_last3m)} in last 3 months')

    # Try last month, but if not enough articles in last month, add another 1 or 2 months
    this_document_type = 'summaries'
    if len(ss_last1m) >= 5:
        this_time_period = f"{one_month_back_date_str}_{current_date_str}"
        this_n_articles = len(ss_last1m)
        this_prompt = get_news_summary_llm_prompt(ss_last1m, 'summary')
        print(f'  Using last month; prompt length ~{len(this_prompt)/4.5:.0f}')
    elif len(ss_last2m) >= 5:
        this_time_period = f"{two_months_back_date_str}_{current_date_str}"
        this_n_articles = len(ss_last2m)
        this_prompt = get_news_summary_llm_prompt(ss_last2m, 'summary')
        print(f'  Using last 2 months; prompt length ~{len(this_prompt)/4.5:.0f}')
    elif len(ss_last3m) >= 1:
        this_time_period = f"{three_months_back_date_str}_{current_date_str}"
        this_n_articles = len(ss_last3m)
        this_prompt = get_news_summary_llm_prompt(ss_last3m, 'summary')
        print(f'  Using last 3 months; prompt length ~{len(this_prompt)/4.5:.0f}')
    else:
        print(f'  No articles in last 3 months')
        this_time_period = None
        this_n_articles = 0
        this_document_type = None

    # Send prompt to OpenAI API to get summary
    if this_n_articles > 0:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': this_prompt}],
                max_tokens=500,
                n=1,
                temperature=0.75
            )
            this_summary = response.choices[0].message.content
        except openai.error.OpenAIError as error:
            print(f'openai {type(error).__name__} for {nn}; skipping')
            this_summary = None
    else:
        this_summary = None

    # for gpt-4o-mini, August 2024
    this_approx_cost_usd = 0 if this_summary is None else (response.usage.prompt_tokens*input_token_cost_usd + response.usage.completion_tokens*output_token_cost_usd)

    res.append(pd.DataFrame({
        'normal_name': nn,
        'create_date': datetime.now(),
        'time_period': this_time_period,
        'n_articles': this_n_articles,
        'document_type': this_document_type,
        'news_coverage_summary': this_summary,
        'summaries_prompt_length_tokens': response.usage.prompt_tokens,
        'approx_cost_usd': this_approx_cost_usd
        }, index=[0]))

    if time.time() - start_time > time_limit_seconds:
        print(f'Out of time in update_news_summaries; quitting at i={i}')
        break

if nn_requiring_update:
    res = pd.concat(res, ignore_index=True)

    # Append to file
    print(f'Generated {len(res)} new news summaries, approximate cost ${res.approx_cost_usd.sum():.2f}')
    pd.concat([news_summaries, res], ignore_index=True).to_feather(news_summaries_path)
