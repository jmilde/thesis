import requests
from tqdm import tqdm
from requests_oauthlib import OAuth1
import urllib.request
from collections import defaultdict
import pandas as pd
import json

def run():
    auth = OAuth1("6ddae208a548427eb559685f98ba0b25", "7a4d8b6b9b5d44f7b0a44e809c591a40")
    #https://www.wordexample.com/list/most-common-verbs-english
    #https://www.wordexample.com/list/most-common-adjectives-english
    # https://www.wordexample.com/list/nouns-with-3-letters  ## manually cleaned

    #df= pd.read_html("https://www.talkenglish.com/vocabulary/top-1500-nouns.aspx")
    #df = df[3][1]
    #df.to_csv("./data/searchwords.csv", sep=";", encoding="utf8")

    df = pd.read_csv("./data/iconDS/searchwords.csv", header=None)
    searchwords = list(set(df[0]))


    labels = defaultdict(lambda : defaultdict())
    for searchword in tqdm(searchwords[:10]):
        endpoint = f"http://api.thenounproject.com/icons/{searchword}?limit_to_public_domain=1&limit=1000"
        response = requests.get(endpoint, auth=auth)
        content  = response.content
        try:
            icons = json.loads(content.decode("utf-8"))["icons"]
            for icon in tqdm(icons):
                img_id = icon["id"]
                img = icon["preview_url"]
                # download img
                urllib.request.urlretrieve(img, f"./data/iconDS/imgs/{img_id}.png")

                term = icon["term"]
                permalink = icon["permalink"]
                link = f"https://thenounproject.com{permalink}"
                # todo get additional tags from website



                labels[img_id]["img_url"] = img
                labels[img_id]["permalink"] = permalink
                labels[img_id]["tags"] = term
                labels[img_id]["searchword"] = searchword
        except json.decoder.JSONDecodeError:
            continue


    data =[[k,*[vv for vv in v.values()]] for k,v in labels.items()]
    df = pd.DataFrame(data, columns=["ID", "img_url", "permalink", "tags", "searchword"])
    df.to_csv("./data/iconDS/labels.csv", sep=";", encoding="utf-8")


if __name__=="__main__":
    run()
