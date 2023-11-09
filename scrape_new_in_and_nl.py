import mechanize, re
# In AWS, install like python3.6 -m pip install mechanize, bs4
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path

#To append to newscatcher file we need 
#  published_date (YYYY-MM-dd HH:MM:SS)
#  link
#  title
#  source = 'irishnews.com' or 'newsletter.co.uk'
#  summary = article text, removing urls (shouldn't be any)
# Then just look for each politician normal_name as lower case in title or summary as lower case
#  and collect as normal_name, published_date, title, link, source, summary
# Dedup: don't allow more than one normal_name-link per day (use short date))

data_dir = './data'
news_articles_filepath = Path(data_dir).joinpath('newscatcher_articles_sep2020topresent.feather')
news_articles_df = pd.read_feather(news_articles_filepath)

politicians = pd.read_csv(Path(data_dir).joinpath('all_politicians_list.csv'))

br = mechanize.Browser()
br.set_handle_robots(False)
br.set_handle_equiv(False)
br.addheaders = [('User-Agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]

# News Letter - just use front page, which goes back about 4 days
# Can't get it to work from AWS instnace
try:
    soup = BeautifulSoup(br.open('https://www.newsletter.co.uk/news/politics').read(), features='html5lib')
    # From politics front page, get links to articles, excluding opinion - gives ~40 stories
    links = [x.attrs['href'] for x in soup.find_all('a', {'class': 'article-title'}) if 'opinion/' not in x['href']]
    links = ['https://www.newsletter.co.uk' + link for link in links]
    nl_res = []
    for link in links:
        soup = BeautifulSoup(br.open(link).read(), features='html5lib')
        title = soup.find('h1').text
        published_date = soup.find('span', {'class': 'published-date-ago'}).text
        published_date = re.sub('Published|th|nd|rd|BST|GMT', '', published_date)
        published_date = re.sub('1st', '1', published_date)  # avoid affecting August too
        published_date = datetime.strptime(published_date.strip(), '%d %b %Y, %H:%M').strftime('%Y-%m-%d %H:%M:%S')
        summary = ' '.join([p.text for p in soup.find('div', {'class':'article-content'}).find_all('div', {'class': 'Markup__ParagraphWrapper-sc-13q6ywe-0'})])
        nl_res.append(pd.DataFrame({'title': [title],
            'published_date': [published_date],
            'link': [link],
            'source': ['newsletter.co.uk'],
            'summary': [summary]}))
    nl_res = pd.concat(nl_res, ignore_index=True)
except:
    nl_res = pd.DataFrame(columns=['title', 'published_date', 'link', 'source', 'summary'])

# Irish News - first sentence only due to paywall; go back max 1 month
soup = BeautifulSoup(br.open('https://www.irishnews.com/news/politicalnews/list/page/1/').read(), features='html5lib')
articles = soup.find_all('article')
articles = [a for a in articles if a.find('time') and datetime.strptime(a.find('time').text, '%d %b, %Y') > datetime.today() - timedelta(days=30)]
# Some political articles don't go to the politics page so check NI page too
soup = BeautifulSoup(br.open('https://www.irishnews.com/news/northernirelandnews/').read(), features='html5lib')
articles2 = soup.find_all('article')
articles2 = [a for a in articles2 if a.find('time') and datetime.strptime(a.find('time').text, '%d %b, %Y') > datetime.today() - timedelta(days=30)]
articles += articles2

in_res = []
for article in articles:
    title = article.find('h1').text.strip() # title
    subtitle = article.find('div', {'class': 'lancio-text'}).text.strip()  # subtitle - might as well append to article stub
    link = article.find('a').attrs['href'] # link
    soup = BeautifulSoup(br.open(link).read(), features='html5lib')
    published_date = soup.find('time').text.strip()  # article date
    published_date = datetime.strptime(published_date, '%d %B, %Y %H:%M').strftime('%Y-%m-%d %H:%M:%S')
    summary = ' '.join([p.text for p in soup.find('div', {'class': 'lancio-text'}).find_all('p')])
    in_res.append(pd.DataFrame({'title': [title],
        'published_date': [published_date],
        'link': [link],
        'source': ['irishnews.com'],
        'summary': '---'.join([subtitle, summary])}))
in_res = pd.concat(in_res, ignore_index=True)

br.close()

res = pd.concat([nl_res, in_res])

mentions = []
for p in politicians.normal_name.unique():
    for _, a in res.iterrows():
        if a.summary.lower().find(p.lower()) > -1 or a.title.lower().find(p.lower()) > -1:
            mentions.append(pd.DataFrame({'normal_name': [p],
                'published_date': [a.published_date],
                'title': [a.title],
                'link': [a.link],
                'source': [a.source],
                'summary': [a.summary]}))
mentions = pd.concat(mentions)
mentions = (mentions
    .assign(short_date = pd.to_datetime(mentions.published_date).dt.strftime('%Y-%m-%d'))
    .drop_duplicates(subset=['normal_name', 'short_date', 'link'])
    .drop(columns=['short_date'])
)

print(f'{len(mentions)} possible new news article mentions')
updated_news_articles_df = pd.concat([news_articles_df, mentions]).drop_duplicates(ignore_index=True)
print(f'Total news article mentions before was {len(news_articles_df)}, and after append+dedup is {len(updated_news_articles_df)}')

if len(updated_news_articles_df) > len(news_articles_df):
    updated_news_articles_df.to_feather(news_articles_filepath, version=1)
    #updated_news_articles_df.to_feather('tmp_news_out.feather', version=1)  # must have v1 to read in R