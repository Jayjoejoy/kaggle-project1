# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 20:39:15 2018

@author: Shaoqing
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import gensim
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.text import Text
from collections import Counter
from time import sleep
import xgboost as xgb
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn import preprocessing
import multiprocessing
from multiprocessing import Pool
import time;
from sklearn.cross_validation import train_test_split

num_partitions = 20
num_cores = multiprocessing.cpu_count()

def parallelize_dataframe(df, func):
    a = np.array_split(df, 20)
    #pool = Pool(num_cores)
    #df = pd.concat(pool.map(func, a))
    #pool.close()
    #pool.join()
    with Pool(12) as pool:
        results = pd.concat(pool.map(func, a))
    return results

def process_essay(df):
    for i in range(np.shape(df)[0]):
        word=[];
        word.extend(word_tokenize(df.iloc[i]))
        word_lower=[m.lower() for m in word]
        english_stopwords = stopwords.words("english")
        english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*', '...']
        words_clear=[]
        for k in word_lower:
            if k not in english_stopwords:
                if k not in english_punctuations:
                    words_clear.append(k)
        st = PorterStemmer()
        words_stem=[st.stem(word) for word in words_clear]
        df.iloc[i]=' '.join(words_stem)
    return df;

if __name__ == '__main__':
    train_data=pd.read_csv('train.csv')
    test_data=pd.read_csv('test.csv')
    meta_data=pd.read_csv('resources.csv')
    
    #summarize the total budget for each proposal
    meta_data['total_price']=meta_data['quantity']*meta_data['price']
    grouped=meta_data.groupby('id')
    temp=grouped.agg('sum'); #let's dismiss the description of resource first
    temp['id']=temp.index
    train_all=train_data.merge(temp,left_on="id",right_on='id') #merge total price columns into train_data
    test_all=test_data.merge(temp,left_on="id",right_on='id') #merge total price columns into test_data
    
    #merge essay columns into one column
    train_all['essay_summary'] = train_all[['project_essay_1','project_essay_2','project_essay_3','project_essay_4']].apply(lambda x : '{} {} {}'.format(x[0],x[1],x[2]), axis=1)
    test_all['essay_summary']  = test_all[['project_essay_1','project_essay_2','project_essay_3','project_essay_4']].apply(lambda x : '{} {} {}'.format(x[0],x[1],x[2]), axis=1)
    
    train_test = train_all.append(test_all);
    train_test=train_test.drop(['id','teacher_id', 'teacher_prefix', 'project_submitted_datetime', \
                                'project_essay_1','project_essay_2', \
                                'project_essay_3','project_essay_4','quantity', 'price'],axis=1)
    
    #pre-process train_data essay_summary, taking about 1 hour and half if without paralization
    df=train_test['essay_summary']
    df.columns=['essay_summary']
    time_start=time.time();
    train_test['essay_summary'] = parallelize_dataframe(df, process_essay)
    time_end=time.time();
    print (time_end-time_start) #take about 2 mins with parallelization
    
    #pre-process train_data project_title
    df=train_test['project_title']
    df.columns=['project_title']
    time_start=time.time();
    train_test['project_title'] = parallelize_dataframe(df, process_essay)
    time_end=time.time();
    print (time_end-time_start)
    
    #pre-process train_data resource summary
    df=train_test['project_resource_summary']
    df.columns=['project_resource_summary']
    time_start=time.time();
    train_test['project_resource_summary'] = parallelize_dataframe(df, process_essay)
    time_end=time.time();
    print (time_end-time_start)
    
    
    #append test data to train data for label processing
    y_train=train_test.iloc[0:182080,2];
    y_train=np.array(y_train);
    print(np.shape(y_train))
    train_test=train_test.drop(['project_is_approved'],axis=1);
    
    #extract label columns
    labels=['project_grade_category','project_subject_categories','project_subject_subcategories','school_state']
    for i in range(4):
        trans=preprocessing.LabelEncoder()
        trans.fit(train_test[labels[i]]);
        train_test[labels[i]]=trans.transform(train_test[labels[i]])
    
    #TF_IDF transform
    tfidf = TfidfVectorizer(max_features=400,norm='l2')
    tfidf.fit(train_test['project_title'])
    project_title=np.asmatrix(tfidf.transform(train_test['project_title'].values.astype('U')).toarray())
    
    #TF_IDF transform
    tfidf = TfidfVectorizer(max_features=400,norm='l2')
    tfidf.fit(train_test['project_resource_summary'])
    project_resource_summary=np.asmatrix(tfidf.transform(train_test['project_resource_summary'].values.astype('U')).toarray())
    
    #TF_IDF transform
    tfidf = TfidfVectorizer(max_features=4000,norm='l2')
    tfidf.fit(train_test['essay_summary'])
    essay_summary=np.asmatrix(tfidf.transform(train_test['essay_summary'].values.astype('U')).toarray())
    
    label_code=train_test[labels].as_matrix()
    x_train=np.concatenate((label_code[0:182080,:],project_title[0:182080,:],project_resource_summary[0:182080,:],essay_summary[0:182080,:]),axis=1)
    x_test=np.concatenate((label_code[182080:260115,:],project_title[182080:260115,:],project_resource_summary[182080:260115,:],essay_summary[182080:260115,:]),axis=1)
    
    x_train.tofile('x_train.bin');
    x_test.tofile('x_test.bin');
    y_train.tofile('y_train.bin');
    #load these binary file
    #x_train=np.fromfile('x_train.bin',dtype=np.float64);
    #x_train.shape=182080,4804;
    #y_train=np.fromfile('y_train.bin',dtype=np.float64);
    #x_test=np.fromfile('x_test.bin',dtype=np.float64);
    #x_test.shape=78035,4804;
    
    train = pd.DataFrame(x_train);
    x_test = pd.DataFrame(x_test);
    y_train = pd.DataFrame(y_train,columns=['label']);
    train['label'] = y_train;
    
    train_xy,val = train_test_split(train, test_size = 0.3,random_state=1)
    y = train_xy.label
    X = train_xy.drop(['label'],axis=1)
    val_y = val.label
    val_X = val.drop(['label'],axis=1)

    xgb_val = xgb.DMatrix(val_X,label=val_y)
    xgb_train = xgb.DMatrix(X, label=y)
    xgb_test = xgb.DMatrix(x_test)

    params={'booster':'gbtree','objective': 'binary:logistic','gamma':0.1,'max_depth':12,'lambda':2,'subsample':0.7,'colsample_bytree':0.7,'min_child_weight':3, 'silent':0 ,'eta': 0.007,'seed':1000,'nthread':7}
    plst = params.items()
    num_round=100;
    watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]
    bst = xgb.train(plst, xgb_train, num_round, watchlist)

    