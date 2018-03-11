import pandas as pd
import numpy as np
# from gensim.models import doc2vec
from gensim import models
from gensim.models.deprecated.doc2vec import LabeledSentence


# Filepath to main training dataset.
train_file_path = "/Users/zhewang/Documents/kaggle/train.csv"
dtype = {
    'id': str,
    'teacher_id': str,
    'teacher_prefix': str,
    'school_state': str,
    'project_submitted_datetime': str,
    'project_grade_category': str,
    'project_subject_categories': str,
    'project_subject_subcategories': str,
    'project_title': str,
    'project_essay_1': str,
    'project_essay_2': str,
    'project_essay_3': str,
    'project_essay_4': str,
    'project_resource_summary': str,
    'teacher_number_of_previously_posted_projects': int,
    'project_is_approved': np.uint8,
}
# Read data and store in DataFrame.
train_data = pd.read_csv(train_file_path, sep=',', dtype=dtype, low_memory=True).sample(10000)
essay1 = train_data['project_essay_1']
ids = train_data['id']

ess1_list = []
for index, row in train_data.iterrows():
	ess1_list.append(LabeledSentence(row['project_essay_1'].split(" "), [row['id']]))
#size is the vector length, window means how many words are included in one paragraph    
model = models.Doc2Vec(size = 100, window = 200, min_count = 3, workers = 1)
vocab = model.build_vocab(ess1_list)
model.train(ess1_list, epochs=10, total_words=100)
# model_loaded = models.Doc2Vec.load('ess1_model.doc2vec')
# print "the first vector is: "
# print model.docvecs[0]
