import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from bs4 import Tag
import time
import re
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
        extracted_text = p.get_text(separator=" ", strip=True)
        if "Translator" in extracted_text:
            index_T = i + 1 #get Translator tag index + 1
            break

        strong_tag = p.find("strong")
        if strong_tag and "Chapter" in strong_tag.get_text(separator=" ", strip=True):
            index_C = i #get Chapter tag index + 1 - 1 because the <div> is counted here meanwhile it won't be counted in findAll('p'),
                        #so we will subtract 1 from index_C = i + 1 because of <div> that won't appear in findAll('p') 

    result = content_div.findAll('p')
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

    chapter_list = []
    num_attempt = 1
    while not chapter_list:
        print(f"get list of chapters attempt {num_attempt}")
        response = requests.get(url)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, "html.parser")
        
        chapter_list = soup.select("div.row")
        if not chapter_list:
            print("Couldn't find chapters on {url}")
            num_attempt += 1

    chapter_links = []
    for chp_list in chapter_list:
        for li in chp_list.find_all("li"):
            a_tag = li.find("a")
            if a_tag and a_tag.has_attr("href"):
                href = a_tag.get("href")
                full_url = base_url + href
                chapter_links.append(full_url)

    chapter_links = [link for link in chapter_links if "lord-of-the-mysteries" in link]

    chp_list = []
    for chp in chapter_links:
        num_attempt = 1
        print(f"Fetching attempt {num_attempt}:  {chp}")
        chp_text = extract_chapter_text(chp)

        while chp_text == "" and num_attempt <= 10: #repeat attempt to access website if didn't work
            num_attempt += 1 
            print(f"Fetching attempt {num_attempt}: {chp}")
            time.sleep(1)
            chp_text = extract_chapter_text(chp)

        if chp_text:
            chp_list.append(chp_text)
    
    return chp_list


def extract_chapter_dragneelclub(chapter_url):
    resp = requests.get(chapter_url)
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, "html.parser")

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
                if tag_type == "h2":
                    result.append([f"##{text}##\n"])
                elif tag_type == "p":
                    result[-1].append(text)

    summary = [[l[0]," ".join(l[1:])] for l in result]
    return summary

def extract_summary_from_dragneelclub(url):
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
df.to_json('data/lotm_dataset', index=False)

