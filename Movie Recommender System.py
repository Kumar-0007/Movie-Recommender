#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head(1)


# In[4]:


credits.head(1)


# In[5]:


movies = movies.merge(credits, on='title')


# In[6]:


movies.head(1)


# In[7]:


movies.info()


# In[8]:


#genres, id, keywords, title, overview, cast(first 3), crew(director)


# In[9]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[10]:


movies.head(1)


# In[11]:


movies.isnull().sum()


# In[12]:


movies.dropna(inplace=True)


# In[13]:


movies.duplicated().sum()


# In[14]:


import ast


# In[15]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[16]:


movies['genres'] = movies['genres'].apply(convert)


# In[17]:


movies.head(1)


# In[18]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[19]:


movies.head(1)


# In[20]:


def convert2(obj):
    L=[]
    cnt=0
    for i in ast.literal_eval(obj):
        if cnt!=3:
            L.append(i['name'])
            cnt+=1
        else:
            break
    return L


# In[21]:


movies['cast'] = movies['cast'].apply(convert2)


# In[22]:


movies.head(1)


# In[23]:


def fetch_dir(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


# In[24]:


movies['crew'] = movies['crew'].apply(fetch_dir)


# In[25]:


movies.head(1)


# In[26]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[27]:


movies.head(1)


# In[28]:


movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])


# In[29]:


movies.head(1)


# In[30]:


movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])


# In[31]:


movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])


# In[32]:


movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])


# In[33]:


movies.head(1)


# In[34]:


movies['tags'] = movies['overview']+ movies['genres']+ movies['keywords']+ movies['cast']+ movies['crew']


# In[35]:


new_df= movies[['movie_id', 'title', 'tags']]


# In[36]:


new_df.head()


# In[37]:


new_df['tags'][0]


# In[38]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[39]:


new_df['tags'][1]


# In[40]:


new_df.head()


# In[41]:


new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# In[42]:


new_df.head()


# In[43]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000, stop_words='english')


# In[44]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[45]:


vectors


# In[46]:


cv.get_feature_names_out()


# In[47]:


import nltk


# In[48]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[49]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[50]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[51]:


new_df.head(1)


# In[52]:


new_df['tags'][0]


# In[53]:


from sklearn.metrics.pairwise import cosine_similarity


# In[54]:


similarity = cosine_similarity(vectors)


# In[55]:


similarity


# In[56]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x: x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# In[57]:


recommend('Batman Begins')


# In[58]:


import pickle


# In[60]:


recommend('Batman')


# In[61]:


recommend('Inception')


# In[62]:


recommend('Spectre')


# In[63]:


recommend('Krrish')


# In[64]:


pickle.dump(new_df.to_dict(), open('movie_dict.pkl','wb'))


# In[65]:


pickle.dump(similarity, open('similarity','wb'))


# In[ ]:




