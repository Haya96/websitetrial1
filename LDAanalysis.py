#@import "D:/dhtmlxSuite_v51_std/codebase/dhtmlxgantt.css"
#import 'dhtmlx-gantt';
# coding: utf-8

# In[2]:


import pandas as pd

data = pd.read_csv('C:/Users/Dummy/Desktop/dataset_2.txt', error_bad_lines=False);
data_text = data[['text']]
data_text['index'] = data_text.index
documents = data_text


# In[3]:


len(documents)


# In[4]:


documents[:5]


# ### Data Preprocessing

# In[5]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)


# In[6]:


import nltk
nltk.download('wordnet')


# #### Lemmatize example

# In[7]:


print(WordNetLemmatizer().lemmatize('went', pos='v'))


# #### Stemmer Example

# In[9]:


stemmer = SnowballStemmer('english')
original_words = ['caresses', 'flies', 'dies', 'mules', 'denied','died', 'agreed', 'owned', 
           'humbled', 'sized','meeting', 'stating', 'siezing', 'itemization','sensational', 
           'traditional', 'reference', 'colonizer','plotted']
singles = [stemmer.stem(plural) for plural in original_words]
pd.DataFrame(data = {'original word': original_words, 'stemmed': singles})


# In[10]:


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# In[36]:


doc_sample = documents[documents['index'] == 16434].values[0][0]

print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))


# In[30]:


processed_docs = documents['text'].map(preprocess)


# In[17]:


processed_docs[:10]


# ### Bag of words on the dataset

# In[32]:


dictionary = gensim.corpora.Dictionary(processed_docs)


# In[33]:


count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# In[34]:


dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)


# In[38]:


bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus
# bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
# bow_corpus


# In[41]:


# bow_doc_4310 = bow_corpus

# for i in range(len(bow_doc_4310)):
#     print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
#                                                      dictionary[bow_doc_4310[i][0]], 
#                                                      bow_doc_4310[i][1]))
from gensim import models
# train the model
tfidf = models.TfidfModel(bow_corpus)
# transform the "system minors" string
tfidf[dictionary.doc2bow("system minors".lower().split())]


# ### TF-IDF

# In[42]:


# from gensim import corpora, models

# tfidf = models.TfidfModel(bow_corpus)


# In[45]:


corpus_tfidf = tfidf[bow_corpus]


# In[46]:


from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break


# ### Running LDA using Bag of Words

# In[47]:


lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)


# In[48]:


for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# Cool! Can you distinguish different topics using the words in each topic and their corresponding weights?

# ### Running LDA using TF-IDF

# In[50]:


lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)


# In[51]:


for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


# ### Classification of the topics

# ### Performance evaluation by classifying sample document using LDA Bag of Words model

# In[53]:


processed_docs


# In[60]:


# for index, score in sorted(lda_model[bow_corpus], key=lambda tup: -1*tup[1]):
#     print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
lda_model[bow_corpus]


# In[79]:


for index, score in sorted(lda_model[bow_corpus], key=lambda tup: -1*tup[0]):
    print("\nScore: {}\t \nTopic: {}".format(score , lda_model.print_topic(index , 10)))


# Our test document has the highest probability to be part of the topic on the top.

# ### Performance evaluation by classifying sample document using LDA TF-IDF model

# In[78]:


for index, score in sorted(lda_model_tfidf[bow_corpus], key=lambda tup: -1*tup[0]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))


# Our test document has the highest probability to be part of the topic on the top.

# ### Testing model on unseen document

# In[82]:


unseen_document = ' The last business trip  I drove to San Francisco  I went to Hertz Rentals and got a 1999 Ford Taurus  thinking it looked comfortable and professional  I found the seating to be uncomfortable for myself  as well as for my passenger Now  seating comfort may not be important to you  but it is to me The fuel usage was fine  the car did get us there with no problems  but  it was such an uncomfortable ride for both of us  It is not as though I am hard to fit into a car'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))

############################
    
 #   with open("D:/python files/jsonn.txt", "w") as f:
 #   json.dump(someResults, f, indent=4)
     
     
#class Foo(object):
 #   def __init__(self):
   #     self.x = lda_model.print_topic(index, 5)
       

#foo = Foo()
#s = json.dumps(foo) # raises TypeError with "is not JSON serializable"

#s = json.dumps(foo.__dict__) # s set to: {"x": system topic}

#if __name__ == '__main__':

 #   a = head_tail()
 #  b = data_info_for_analysis()
 #  c = data_visualization_chart()
 #  d = missing_values_duplicates()
 #  e = mapping_yes_no()
 #   f = one_hot_encoding()
 #   g = outlier_identification()

 #   out2 = removing_outliers()
 #   h = droping, features = removing_unwanted_columns(out2)

 #   df_telecom_test, df_telecom_train, probs, clf = random_model_predictions(droping, features)

 #   i = logistic_model_prediction(df_telecom_train, df_telecom_test, features)
 #   j = decision_model_prediction(df_telecom_train, df_telecom_test, features)

 #   k = fpr_tpr_thresholds(df_telecom_test, probs, clf, features)

