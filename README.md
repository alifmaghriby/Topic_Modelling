#TOPIC_MODELLING_INDONESIA_LANGUAGE

!pip install --upgrade gensim

!pip install pyldavis==3.2.1

!pip install Sastrawi

!pip install swifter

# Install Library
import pandas as pd
import numpy as np
import nltk

# Upload Data
df = pd.read_excel('file_name.xlsx')
df

# Rename a single column
df = df.rename(columns={'old_name': 'new_name'})

# Case Folding
df['column_name'] = df['column_name'].str.lower()
print('Case Folding Result : \n')
print(df['column_name'].head(5))

# Replace missing values with a specific value
df = df.fillna('null')

#Re-Check for Missing Values after Replace Missing Values
df.isnull().sum()

# Library untuk Tokenizing
import string
import re #regex library

# Import word_tokenizing & FreqDist from NLTK
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# ------ Tokenizing ---------

nltk.download('punkt')

def remove_tweet_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")

df['column_name'] = df['column_name'].apply(remove_tweet_special)

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

df['column_name'] = df['column_name'].apply(remove_number)

#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

df['column_name'] = df['column_name'].apply(remove_punctuation)

#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

df['column_name'] = df['column_name'].apply(remove_whitespace_LT)

#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

df['column_name'] = df['column_name'].apply(remove_whitespace_multiple)

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

df['column_name'] = df['column_name'].apply(remove_singl_char)

# NLTK word tokenize
def word_tokenize_wrapper(text):
    return word_tokenize(text)

df['column_name_tokens'] = df['column_name'].apply(word_tokenize_wrapper)

print('Tokenizing Result : \n')
print(df['column_name_tokens'].head())

# Stopwords
nltk.download('stopwords')

from nltk.corpus import stopwords

list_stopwords = stopwords.words('indonesian')

# remove stopwords manually
list_stopwords.extend(["yg","kl","kalo","dgn","jd","utk","jgn","si","smua","nya"])

# remove stopwords pada list token
def stopwords_removal(words):
  return[word for word in words if word not in list_stopwords]

df['column_name_tokens_WSW'] = df['column_name_tokens'].apply(stopwords_removal)

print(df['column_name_tokens_WSW'].head())

# Normalisasi

normalized_word = pd.read_excel('/content/normalisasi.xlsx')

normalized_word_dict = {}

for index, row in normalized_word.iterrows():
  if row[0] not in normalized_word_dict:
    normalized_word_dict[row[0]] = row[1]

  def normalized_term(document):
    return [normalized_word_dict[term] if term in normalized_word_dict else term for term in document]

df['column_name_normalized'] = df['column_namee_tokens_WSW'].apply(normalized_term)
df['column_name_normalized'].head(10)

# Import Sastrawi Package

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
  return stemmer.stem(term)

term_dict = {}

for document in df['column_name_normalized']:
  for term in document:
    if term not in term_dict:
      term_dict[term]=''

print(len(term_dict))

for term in term_dict:
  term_dict[term] = stemmed_wrapper(term)

# Show Results
  print(term,':',term_dict[term])

  # Apply stemmed term to dataframe
def get_stemmed_term(document):
  return [term_dict[term] for term in document]

df['column_name_tokens_stemmed'] = df['column_name_normalized'].swifter.apply(get_stemmed_term)

print(df['column_name_tokens_stemmed'])

# Stopword # 2

from nltk.corpus import stopwords

list_stopwords = stopwords.words('indonesian')

# Convert List to Dictionary
list_stopwords = set(list_stopwords)

# Remove Stopwords Pada List Token
def stopwords_removal(words):
  return [word for word in words if word not in list_stopwords]

df['column_name_tokens_stemmed2'] = df['column_name_tokens_stemmed'].apply(stopwords_removal)

print(df['column_name_tokens_stemmed2'].head())

for i in range(len(df)):
  a=df.iloc[i][10]
  document.append(a)

document[0:10]

doc_clean = df['column_name_tokens_stemmed2']
doc_clean

**#LDA**
import gensim
from gensim import corpora

dictionary = corpora.Dictionary(doc_clean)
print(dictionary)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the Object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

total_topics = 10 #number of topic to extract
number_words = 10 #number of words per topic

# Running and Training LDA Model on the document term matrix
lda_model = Lda(doc_term_matrix, num_topics=total_topics, id2word = dictionary, passes=20)
lda_model.show_topics(num_topics=total_topics, num_words=number_words)

# Word Count of Topic Keywords
from collections import Counter
topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in doc_clean for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
  for word, weight in topic:
    out.append([word, i, weight, counter[word]])

df_imp_wcount = pd.DataFrame(out, columns=['word','topic_id','importance','word_count'])
print(df_imp_wcount)

#SAVE
file_name = 'file_name.xlsx'
df_imp_wcount.to_excel(file_name, index = False, header=True)
