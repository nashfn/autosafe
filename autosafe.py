import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

client=OpenAI()

NV_OUTFILE = "nvidia_stock.pickle"
NV_EMBED_OUTFILE = "nvidia_embed.pickle"

load_dotenv()

# routine to flag 
# Steps:
# 1. read the pickle file, 
# 2. given a string, compute embedding, cosine similarity, create threshold and flag.

def oai_embedding(input):
    # print("oai_embedding, input = ", input)
    x = client.embeddings.create(
      model="text-embedding-ada-002",
      input=input,
      encoding_format="float")
    return x.data[0].embedding

def download_url(url_string):
    html_content = requests.get(url_string).text
    soup = BeautifulSoup(html_content, 'html.parser') # 'can also use html.parser, lxml etc'
    paragraphs = soup.find_all('p')
    full_text = [p.text for p in paragraphs]
    print("number of paragraphs: ", len(full_text))
    return full_text

def run_oai(system_msg, model_str, user_query):
    completion = client.chat.completions.create(
        model=model_str,
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": user_query
            }
        ]
    )
    return completion.choices[0].message


def compute_cos(X, Y):
    # Convert embeddings to a numpy array
    np_X = np.array(X)
    np_Y = np.array(Y)

    # Compute cosine similarity between the embeddings
    similarity_matrix = cosine_similarity(np_X, np_Y)
    return similarity_matrix

with open(NV_OUTFILE, "rb") as infile:
    data_pairs = pickle.load(infile)

with open(NV_EMBED_OUTFILE, "rb") as infile: 
    nv_embed = pickle.load(infile)


def autosafe_filter(msg, nv_embed):
    msg_embed = oai_embedding(msg)
    msg_sim = compute_cos(nv_embed, [msg_embed])
    # print(msg_sim)
    return (np.mean(msg_sim[0]) > 0.72)