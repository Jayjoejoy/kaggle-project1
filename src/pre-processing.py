import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gensim
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.text import Text
from collections import Counter

train_data=pd.read_csv('train.csv')
predictors=list(train_data.columns)
predictors.remove('teacher_id')
predictors.remove('teacher_prefix')
predictors.remove('project_submitted_datetime')

plt.hist(train_data['project_is_approved'])
plt.hist(train_data['school_state'],bins=51,align=u'mid',rwidth=1)
plt.hist(train_data['project_grade_category'])
plt.hist(train_data['project_subject_categories'],bins=51,align=u'mid',rwidth=1)

meta_data=pd.read_csv('resources.csv')
meta_cols=list(meta_data.columns)

# I just checked the essay_1 part
fid=open('essay_1.txt','w');
for i in range(np.shape(train_data)[0]):
    fid.write(train_data['project_essay_1'][i])
fid.close()

# split paragraphs to sentence
sent=[];
for i in range(np.shape(train_data)[0]):
    sent.append(sent_tokenize(train_data['project_essay_1'][i]))
# split sentence into words
word=[];
for i in sent:
    for j in i:
        word.extend(word_tokenize(j))
# Capital words to lower words
word_lower=[i.lower() for i in word]
# discard unuseful words
english_stopwords = stopwords.words("english")
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*', '...']
words_clear=[]
for i in word_lower:
    if i not in english_stopwords:
        if i not in english_punctuations:
            words_clear.append(i)
# get stem words: e.g. running --> run           
st = PorterStemmer()
words_stem=[st.stem(word) for word in words_clear]
# change list to text format
word_text=Text(words_stem)

words_counter=Counter(words_stem)
# find top 20 words
words_counter.most_common(20)
# find top 20 phrases
word_text.collocations(num=20, window_size=2)