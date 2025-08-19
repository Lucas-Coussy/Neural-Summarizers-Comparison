import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
import time
import chromedriver_autoinstaller
from bs4 import Tag
import time
import re
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

url_list = ['https://lotmsummaries.wordpress.com/2024/03/24/chapters-1-10-summary/','https://lotmsummaries.wordpress.com/2024/04/30/chapters-11-20-analysis-and-summary-chapter-by-chapter/',
            'https://lotmsummaries.wordpress.com/2024/06/09/chapters-21-30-summary-and-analysis-combined/','https://lotmsummaries.wordpress.com/2024/06/11/chapters-31-40-summary-and-analysis-combined/',
            'https://lotmsummaries.wordpress.com/2024/06/16/chapters-41-50-summary-and-analysis/','https://lotmsummaries.wordpress.com/2024/06/16/chapters-51-60-summary-and-analysis/',
            'https://lotmsummaries.wordpress.com/2024/06/18/chapter-61-70-summary-and-analysis/']


def extract_summary_from_lotmsummaries(chapter_url):
    resp = requests.get(chapter_url)
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, "html.parser")
    # Placeholder: locate where the summary appears
    content = soup.find("div", class_="entry-content")
    paragraphs = content.find_all("p") if content else []
    summaries = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
    return summaries

def extract_chapter_text(chapter_url):
    resp = requests.get(chapter_url)
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, "html5lib")  #more forgiving than html.parser

    # Locate the container with the class chapter-c
    content_div = soup.select_one("div.chapter-c") 
    if not content_div:
        print(f"Couldn't find text on {chapter_url}")
        return ""
    index_C = 0 #index tag chapter
    index_T = 0 #index tag Translator
    for i,p in enumerate(content_div): 
        #print(i,p)
        extracted_text = p.get_text(separator=" ", strip=True)
        if "Translator" in extracted_text:
            index_T = i + 1 #get Translator tag index + 1
            break

        strong_tag = p.find("strong")
        if strong_tag and "Chapter" in strong_tag.get_text(separator=" ", strip=True):
            index_C = i + 1 #get Chapter tag index + 1
    
    result = content_div.findAll('p')
    #print(result)
    if index_T != 0:
        text = "\n".join(p.get_text(strip=True) for p in result[index_T:] if p.get_text(strip=True))
    else:
        text = "\n".join(p.get_text(strip=True) for p in result[index_C:] if p.get_text(strip=True))

    text = text.replace('“', '"').replace('”', '"').replace("’","'").replace("—","-").replace('…', '...')
    return [text]

import requests
from bs4 import BeautifulSoup

def extract_text_from_novelfull(url):
    base_url = 'https://novelfull.net'

    response = requests.get(url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, "html.parser")
    
    chapter_list = soup.select("div.row")
    if not chapter_list:
        print("❌ Could not find div.row")
        return []

    chapter_links = []
    for chp_list in chapter_list:
        for li in chp_list.find_all("li"):
            a_tag = li.find("a")
            if a_tag and a_tag.has_attr("href"):
                href = a_tag.get("href")
                full_url = base_url + href
                chapter_links.append(full_url)

    chapter_links = chapter_links

    chp_list = []
    for chp in chapter_links:
        print(f"Fetching: {chp}")
        #time.sleep(1)
        chp_text = extract_chapter_text(chp)
        if chp_text:
            chp_list.append(chp_text)
    
    return chp_list



summary_list = []
for url in url_list:
    summaries = extract_summary_from_lotmsummaries(url)[0:10]
    summary_list.extend(summaries)

#text = extract_text_from_novelfull("https://novelfull.net/lord-of-the-mysteries.html")
#text2 = extract_summary_from_novelfull('https://novelfull.net/lord-of-the-mysteries.html?page=2')

"""
text = [[""]] + text
text_f = text + text2
print(len(text_f))
print(len(summary_list))

df = pd.DataFrame(text_f[0:70], columns=['text'])

df['summary'] = summary_list
print(df)
df.to_csv('data/lotm_dataset', index=False)
"""

def extract_chapter_dragneelclub(chapter_url):
    resp = requests.get(chapter_url)
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, "html.parser")

    #for tag in soup.find_all("div"):
     #   if tag.has_attr("id"):
      #      print(f"ID: {tag['id']}")
       #     print(tag.prettify())

    # Locate the container with the chapter paragraphs
    content_div = soup.find("div", id="page")
    if not content_div:
        print(f"Could not find text on {chapter_url}")
        return ""

    # Extract and join all <p> texts
    result = []

    for elem in content_div.find_all(["h2", "p"], recursive=True):
        if isinstance(elem, Tag):
            text = elem.get_text(strip=True)
            if text:
                text = text.replace('“', '"').replace('”', '"').replace("’", "'").replace("—", "-").replace('…', '...')
                tag_type = elem.name
                # Optionally format or label it
                if tag_type == "h2":
                    result.append([f"##{text}##\n"])
                elif tag_type == "p":
                    result[-1].append(text)

    summary = [[l[0]," ".join(l[1:])] for l in result]
    return summary

def extract_summary_from_dragneelclub(url):
    base_url = 'https://novelfull.net'

    response = requests.get(url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, "html.parser")
    
    chapter_list = soup.select("div.entry-summary")
    if not chapter_list:
        print("❌ Could not find .list-chapter. Page may be JS-rendered.")
        return []

    chapter_links = []
    for chp_list in chapter_list:
        for p in chp_list.find_all("p"):
            a_tag = p.find("a")
            if a_tag and a_tag.has_attr("href"):
                href = a_tag.get("href")
                full_url = href
                chapter_links.append(full_url)
    print(chapter_links)
    chp_list = []
    for chp in chapter_links:
        print(f"Fetching: {chp}")
        chp_text = extract_chapter_dragneelclub(chp)
        if chp_text:
            chp_list.append(chp_text)
    summary = [chp for group_chp in chp_list for chp in group_chp]        
    return summary

def extract_text_and_summary(num_row : int):
    num_page_summary = (num_row // 90) + 1
    num_page_text = (num_row // 50) + 1
    assert num_page_summary <= 16
    assert num_page_text <= 29
    x = []
    y = []

    x = extract_text_from_novelfull("https://novelfull.net/lord-of-the-mysteries.html")
    for num_page in range(1,num_page_text):
            x = x + extract_text_from_novelfull(f"https://novelfull.net/lord-of-the-mysteries.html?page={num_page+1}")
    for num_page in range(num_page_summary):
        y = y + extract_summary_from_dragneelclub(f"https://dragneelclub.com/category/chapters/lord-of-the-mysteries/page/{16-num_page}/")
    #x = [[""]] + x #the first chp doesn't get downloaded so we add [""]

    y_chp = [int(re.match(r"\d+",summary[0].split()[1]).group()) for summary in y if summary] #get chp list
    y = [y for _,y in sorted(zip(y_chp,y))] #sort y according to chapter value
    df = pd.DataFrame({'text':x})
    x, y = x[0:num_row], y[0:num_row] #truncate supplementary row
    print("len : ",len(x),len(y))
    df = pd.DataFrame({'text':x, 'summary':y})
    return df

df = extract_text_and_summary(1432) #nb of chp
print(df)
#df.to_csv('data/lotm_dataset', index=False)
df.to_json('data/lotm_dataset', index=False)