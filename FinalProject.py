#!/usr/bin/env python
# coding: utf-8

# ## Data Processing

# In[1]:


import nltk
from nltk.corpus import stopwords
import string
from collections import Counter
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd
from nltk.stem import PorterStemmer


# In[2]:


nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


# In[3]:


def normalize(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text


# In[4]:


with open('wikisent2.txt', 'r') as infile, open('cleanedtext.txt', 'w') as outfile:
    for line in infile:
        normalized_line = normalize(line)
        outfile.write(normalized_line + '\n')


# ## Indexing

# In[5]:


def create_inverted_index(text):
    inverted_index = defaultdict(list)
    
    tokens = re.findall(r'\w+', text.lower())

    for position, token in enumerate(tokens):
        inverted_index[token].append(position)  

    return inverted_index


# In[15]:


# Example
with open('cleanedtext.txt', 'r', encoding='utf-8') as file:
    text = file.read()

inverted_index = create_inverted_index(text)

sample_word = 'music'
if sample_word in inverted_index:
    print(f"{sample_word}: {inverted_index[sample_word]}")
else:
    print(f"The word '{sample_word}' is not found in the document.")


# ## Search Algorithm

# In[7]:


with open('cleanedtext.txt', 'r') as file:
    document = file.read()


# In[8]:


word_frequencies = Counter(document.split())


# In[9]:


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([document])
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray()[0]


# In[10]:


df = pd.DataFrame({
    'Word': feature_names,
    'TF-IDF Score': tfidf_scores,
    'Frequency': [word_frequencies.get(word, 0) for word in feature_names]
})

print(df)


# ## Ranking

# In[11]:


with open('cleanedtext.txt', 'r') as file:
    data = file.read().replace('\n', '')


# In[12]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform([data])

df = pd.DataFrame(tfidf_matrix[0].T.todense(), index=tfidf_vectorizer.get_feature_names_out(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)


# In[13]:


print("Top 5 words with the highest TF-IDF scores:")
print(df.head(5))


# In[14]:


words = re.findall(r'\w+', data.lower())
word_freq = Counter(words)
print("\nTop 5 most frequently occurring words:")
for word, freq in word_freq.most_common(5):
    print(f"{word}: {freq}")

