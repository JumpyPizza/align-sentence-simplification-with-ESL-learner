#  a note-book style script to evaluate frequency and diversity
import requests
import json
from tqdm import tqdm
import random
import torch
import numpy as np
import re


complicated_sentence_f = "./data/gpt4_complications_score_wiki_varied.txt"
complicated_score_wiki = []
with open(complicated_sentence_f, "r") as f:
  lines = f.readlines()
  for line in lines:
    sent = json.loads(line)['response']['body']['choices'][0]['message']['content']
    complicated_score_wiki.append(sent)

score_test = "https://raw.githubusercontent.com/yukiar/CEFR-SP/main/CEFR-SP/SCoRE/CEFR-SP_SCoRE_test.txt"
score_train = "https://raw.githubusercontent.com/yukiar/CEFR-SP/main/CEFR-SP/SCoRE/CEFR-SP_SCoRE_train.txt"
score_dev = "https://raw.githubusercontent.com/yukiar/CEFR-SP/main/CEFR-SP/SCoRE/CEFR-SP_SCoRE_dev.txt"

wiki_train = "https://raw.githubusercontent.com/yukiar/CEFR-SP/main/CEFR-SP/Wiki-Auto/CEFR-SP_Wikiauto_train.txt"
wiki_test = "https://raw.githubusercontent.com/yukiar/CEFR-SP/main/CEFR-SP/Wiki-Auto/CEFR-SP_Wikiauto_test.txt"
wiki_dev = "https://raw.githubusercontent.com/yukiar/CEFR-SP/main/CEFR-SP/Wiki-Auto/CEFR-SP_Wikiauto_dev.txt"

score_test_list = requests.get(score_test).text.split("\n")[:-1]
score_train_list = requests.get(score_train).text.split("\n")[:-1]
score_dev_list = requests.get(score_dev).text.split("\n")[:-1]

wiki_test_list = requests.get(wiki_test).text.split("\n")[:-1]
wiki_train_list = requests.get(wiki_train).text.split("\n")[:-1]
wiki_dev_list = requests.get(wiki_dev).text.split("\n")[:-1]

score_test_sentences = [i.split("\t")[0] for i in score_test_list]
score_train_sentences = [i.split("\t")[0] for i in score_train_list]
score_dev_sentences = [i.split("\t")[0] for i in score_dev_list]

wiki_test_sentences = [i.split("\t")[0] for i in wiki_test_list]
wiki_train_sentences = [i.split("\t")[0] for i in wiki_train_list]
wiki_dev_sentences = [i.split("\t")[0] for i in wiki_dev_list]


def get_corresponsing_lines(sentence_lines):
  # Example lengths of the original lists
  length_score_test_list = len(score_test_list)
  length_score_train_list = len(score_train_list)
  length_score_dev_list = len(score_dev_list)
  length_wiki_test_list = len(wiki_test_list)
  length_wiki_train_list = len(wiki_train_list)
  length_wiki_dev_list = len(wiki_dev_list)

  # Reconstructing the lists from sentence_lines
  score_test_list_reconstructed = sentence_lines[:length_score_test_list]
  score_train_list_reconstructed = sentence_lines[length_score_test_list:length_score_test_list + length_score_train_list]
  score_dev_list_reconstructed = sentence_lines[length_score_test_list + length_score_train_list:length_score_test_list + length_score_train_list + length_score_dev_list]
  wiki_test_list_reconstructed = sentence_lines[length_score_test_list + length_score_train_list + length_score_dev_list:length_score_test_list + length_score_train_list + length_score_dev_list + length_wiki_test_list]
  wiki_train_list_reconstructed = sentence_lines[length_score_test_list + length_score_train_list + length_score_dev_list + length_wiki_test_list:length_score_test_list + length_score_train_list + length_score_dev_list + length_wiki_test_list + length_wiki_train_list]
  wiki_dev_list_reconstructed = sentence_lines[length_score_test_list + length_score_train_list + length_score_dev_list + length_wiki_test_list + length_wiki_train_list:]
  return score_test_list_reconstructed,  score_train_list_reconstructed, score_dev_list_reconstructed, wiki_test_list_reconstructed, wiki_train_list_reconstructed, wiki_dev_list_reconstructed

score_test_comp, score_train_comp, score_dev_comp, wiki_test_comp, wiki_train_comp, wiki_dev_comp = get_corresponsing_lines(complicated_score_wiki)

test_comp = score_test_comp+wiki_test_comp
test_ref = score_test_sentences+wiki_test_sentences


##### evaluate word coverage ####
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
df = pd.read_csv("./CEFR-SP/english_profile_words.csv")
A1_df = df[df['level'] == 'A1']
A2_df = df[df['level'] == 'A2']
A_base_words = A1_df.base.to_list() + A2_df.base.to_list()
B1_df = df[df['level'] == 'B1']
B2_df = df[df['level'] == 'B2']
B_base_words = B1_df.base.to_list() + B2_df.base.to_list()
C1_df = df[df['level'] == 'C1']
C2_df = df[df['level'] == 'C2']
C_base_words = C1_df.base.to_list() + C2_df.base.to_list()


def process_word(word):
  word = word.strip()
  if len(word.split(" "))>1: # is a phrase
    # 1. remove ", etc.", "be"
    phrase = word.replace(", etc.", "")
    phrase = phrase.replace("be", "")
    phrase = phrase.replace(" to do sth", "")
    # remove anything in ()
    phrase = re.sub(r'\(.*?\)', '', phrase)
    # remove "be"
    # 2. split with " or "
    phrase = phrase.split(" or ")[:1]
    # 3. deal with sb/sth
    for idx, p in enumerate(phrase):
      p = p.replace("sb/sth", "...")
      p = p.replace("sb", "...")
      p = p.replace("sth", "...")
      phrase[idx] = p
    # 4. deal with / and ()
    for idx, p in enumerate(phrase):
      if "/" in p:
        new_p = []
        p_parts = p.split(" ")
        for part in p_parts:
          if "/" in part:
            part = part.split("/")[0]
          new_p.append(part)

        phrase[idx] = " ".join(new_p)
    # 5. deal with ...
    new_phrase = []
    for p in phrase:
      if "..." in p:
        p = p.split("...")
        for splitted in p:
          new_phrase.append(splitted)
      else:
        new_phrase.append(p)
    new_phrase = [p.lstrip().rstrip() for p in new_phrase]
    new_phrase = [re.sub(r'\s+', ' ', p).lower() for p in new_phrase if len(p)>0] # replace multiple blank
    phrase_no_redundant = []
    for p in new_phrase:
      if p not in phrase_no_redundant:
        phrase_no_redundant.append(p)

    return phrase_no_redundant
  else:
    return word.lower()


# 1. Tokenize and remove stop words
# 2. Removing Punctuation
# 3. Normalization: case and lemma
# 4. partial match: one exact match is 1, and partial is divided by the matched num of char


lemmatizer = WordNetLemmatizer()
# Function to preprocess and lemmatize words
def preprocess(text):
    # works only for English
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    # convert to lowercase
    tokens = [word.lower() for word in tokens]
    # Remove p\unctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    # tokens = [word.lower() for word in tokens if word.isalnum()]
    # Remove stopwords
    no_stop_tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    with_stop_tokens = tokens
    lemmatized_tokens_no_stop = [lemmatizer.lemmatize(token) for token in no_stop_tokens]
    lemmatized_tokens_stop = [lemmatizer.lemmatize(token) for token in with_stop_tokens] # stop tokens are included
    return lemmatized_tokens_no_stop, lemmatized_tokens_stop

def count_occurrence(sentence_list, word_list):
    # TODO: improve match algorithm
    word_freq_dict = {}
    for idx, w in enumerate(word_list):
      word_freq_dict[idx] = 0
    total_tokens = 0
    for sentence_text in sentence_list:
      sentence_tokens_no_stop, sentence_tokens_stop = preprocess(sentence_text)
      sentence_token_num = len(sentence_tokens_no_stop)
      total_tokens += sentence_token_num
      # search for words
      for token in sentence_tokens_no_stop:
        for idx, w in enumerate(word_list):
          if isinstance(w, str): # word
            # exactly the same
            if token == w:
              word_freq_dict[idx] += 1
              continue # end when matched

      # search for phrases
      for idx, w in enumerate(word_list):
        if isinstance(w, list): #phrase

          # match continue phrase
          if len(w) == 1: # continue phrase
            if w[0] in " ".join(sentence_tokens_stop):
              word_freq_dict[idx] += 1
          elif len(w) > 1:
            matched_part = 0
            sentence_to_match = " ".join(sentence_tokens_stop)
            for part in w:
              if len(part.split(" "))>1: # we ignore the single word when matching discontinued phrase for simplicity
                if part in sentence_to_match:
                  matched_part = 1
                  sentence_to_match = "".join(sentence_to_match.split(part)[1:]) # only check the following part of sentence after one partial match
            if matched_part == 1:
              word_freq_dict[idx] += 1
    return word_freq_dict, total_tokens

A_word_list = []
for w in A_base_words:
  processed_words = process_word(w)
  if processed_words not in A_word_list and processed_words not in B_base_words and processed_words not in C_base_words:
    A_word_list.append(processed_words)

B_word_list = []
for w in B_base_words:
  processed_words = process_word(w)
  if processed_words not in B_word_list and processed_words not in A_base_words and processed_words not in C_base_words:
    B_word_list.append(processed_words)

C_word_list = []
for w in C_base_words:
  processed_words = process_word(w)
  if processed_words not in C_word_list and processed_words not in A_base_words and processed_words not in B_base_words:
    C_word_list.append(processed_words)


def get_coverage(word_list, sent_list):
  word_freq_dict, total_tokens = count_occurrence(sent_list, word_list)

  total_cnt = sum(word_freq_dict.values())
  cnt_num = 0
  for k, v in word_freq_dict.items():
    if v>0:
      cnt_num += 1
  freq = total_cnt/total_tokens
  diversity = cnt_num/len(word_freq_dict)
  print(f"frequency is {freq}, diversity is {diversity}")

  #### example usage #####
  for word_list in [C_word_list, B_word_list, A_word_list]:

    get_coverage(word_list, test_ref)