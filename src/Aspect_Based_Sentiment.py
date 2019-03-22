
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn import svm
import numpy as np
import nltk
from nltk.corpus import stopwords
import pandas as pd
import pickle
import sys


# In[2]:


window_size = 20
stop = set(stopwords.words('english'))


# In[3]:


def removeSpecialCharacters(text):
    ret = []
    for word in text:
        if(word.isalpha()):
            ret.append(word)
    if(len(ret) == 0):
        return text
    else:
        return ret


# In[4]:


def changeToLower(text):
    return [word.lower() for word in text]


# In[5]:


def removeStopWords(text):
    ret = []
    for word in text:
        if(word not in stop):
            ret.append(word)
    if(len(ret) == 0):
        return text
    else:
        return ret


# In[6]:


def mergePosTaggedText(text):
    ret = ''
    for word in text:
        ret += word + ' '
    ret = ret[:-1]
    return ret


# In[7]:


def cleanText(df):
    df[' text'] = df[' text'].replace(regex=r'\[comma\]', value=',')
    df[' aspect_term'] = df[' aspect_term'].replace(regex=r'\[comma\]', value=',')
    df['text_wt'] = df[' text'].apply(nltk.word_tokenize)
    df['aspect_term_wt'] = df[' aspect_term'].apply(nltk.word_tokenize)
    df['text_wt'] = df['text_wt'].apply(removeSpecialCharacters)
    df['aspect_term_wt'] = df['aspect_term_wt'].apply(removeSpecialCharacters)
    df['text_wt'] = df['text_wt'].apply(changeToLower)
    df['aspect_term_wt'] = df['aspect_term_wt'].apply(changeToLower)
    df['text_wt'] = df['text_wt'].apply(removeStopWords)
    df['aspect_term_wt'] = df['aspect_term_wt'].apply(removeStopWords)
    return df


# In[8]:


def getWindowText(df):
    df = cleanText(df)
    new_text = []
    for index,row in df.iterrows():
        new_row_text = []
        aspect_row = row['aspect_term_wt']
        text = row['text_wt']
        l_text = len(text)

        for index,word in enumerate(text):
            if(word == aspect_row[0]):
                end_left = index
                break

        for index,word in enumerate(text):   
            if(word == aspect_row[-1]):
                start_right = index+1
                break

        start_left = max(0,end_left-window_size)
        end_right = min(l_text,start_right+window_size)
        new_row_text += text[start_left:end_right]
        new_text.append(new_row_text)


    df['text_wt'] = new_text
    return df['text_wt'].apply(mergePosTaggedText)


# In[9]:


def train(path_train_data,model_name):
    df = pd.read_csv(path_train_data,usecols=["example_id", " text"," aspect_term", ' term_location', ' class']) #Read Data
    data = getWindowText(df) #Get Cleaned Text
    labels = df[' class']  #Get Label
    
    vectorizer = CountVectorizer(decode_error="replace") #Count Vectorizer
    transformer = TfidfTransformer()
    data_tfidf = transformer.fit_transform(vectorizer.fit_transform(data))

    clf = svm.LinearSVC(class_weight={-1:0.33,0:0.34,1:0.33},C=0.5) #Classifier
    clf.fit(data_tfidf,labels)
    
    pickle.dump([vectorizer.vocabulary_,clf], open(model_name, 'wb'))


# In[10]:


def test(path_test_data,model_name,path_out_txt):
    df_test = pd.read_csv(path_test_data,usecols=["example_id", " text"," aspect_term", ' term_location'])
    
    data_test = getWindowText(df_test)
    
    vectorizer,clf = pickle.load(open(model_name, 'rb'))
    
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vectorizer)
    data_tfidf = transformer.fit_transform(loaded_vec.fit_transform(data_test))
    
    pred = clf.predict(data_tfidf)
    
    out = ''
    for ex_id,p in zip(df_test['example_id'],pred):
        out += ex_id + ';;' + str(p) + '\n'
    out = out[:-1]
    file = open(path_out_txt,"w+") 
    file.write(out)
    file.close()


# In[11]:


if(sys.argv[1] == '-test' and len(sys.argv) != 5):
    print('Invalid Arguments for Test!')
elif(sys.argv[1] == '-train' and len(sys.argv) != 4):
    print('Invalid Arguments for Train!')
elif sys.argv[1] == '-train':
    train(sys.argv[2],sys.argv[3])
elif sys.argv[1] == '-test':
    test(sys.argv[2],sys.argv[3],sys.argv[4])

