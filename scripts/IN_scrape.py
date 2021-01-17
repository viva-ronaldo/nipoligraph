from bs4 import BeautifulSoup
import requests
import pandas as pd

## don't truncate printed urls
pd.set_option('display.max_colwidth', None)

## spoof user-agent
headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36',
            'Content-Type': 'text/html'}

## category urls  
urls = ['http://www.irishnews.com/dyna/partial/articlesbycategory?s=0&e=2501&category=%2firishnews%2fnews%2fpoliticalnews&onlyPrimary=false',
       'http://www.irishnews.com/dyna/partial/articlesbycategory?s=0&e=2501&category=%2firishnews%2fnews%2frepublicofirelandnews&onlyPrimary=false',
       'http://www.irishnews.com/dyna/partial/articlesbycategory?s=0&e=2501&category=%2firishnews%2fnews%2fassemblyelection&onlyPrimary=false',
       'http://www.irishnews.com/dyna/partial/articlesbycategory?s=0&e=2501&category=%2firishnews%2fnews%2fbrexit&onlyPrimary=false',
       'http://www.irishnews.com/dyna/partial/articlesbycategory?s=0&e=2501&category=%2firishnews%2fnews%2fcivilrights&onlyPrimary=false',
       'http://www.irishnews.com/dyna/partial/articlesbycategory?s=0&e=2501&category=%2firishnews%2fnews%2fgeneralelection&onlyPrimary=false',
       'http://www.irishnews.com/dyna/partial/articlesbycategory?s=0&e=2501&category=%2firishnews%2fnews%2fnorthernirelandnews&onlyPrimary=false',
       'http://www.irishnews.com/dyna/partial/articlesbycategory?s=0&e=2501&category=%2firishnews%2fnews%2fuknews&onlyPrimary=false']

## takes <article></article> element and converts to df row
def elementToRow(soup):
    titleBlock = soup.find(class_="lancio-title")
    title = titleBlock.find("a").get_text()
    url = titleBlock.find("a")["href"].lstrip("/")
    ## warning, stub is often similar to title
    stub = soup.find(class_="lancio-text").get_text()
    if(soup.find(class_="lancio-tag")):
        author = soup.find(class_="lancio-tag").get_text()
    else:
        author = ""    
    rawDate = soup.find(class_="lancio-datetime-string")['datetime'].split(" ")[0]
    d = {"title": [title], "url": [url], "stub":[stub], "author":[author], "date":[rawDate]}
    df = pd.DataFrame(data=d)
    df['date'] = pd.to_datetime(df['date'])
    return df

## iterate over category urls, populate df
def scrape():
    res = pd.DataFrame()
    for u in urls:
        r = requests.get(u, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser');
        for ele in soup.find_all(class_="row lancio"):
            tmp  = [res, elementToRow(ele)]
            res = pd.concat(tmp)
    return res

print(scrape())
