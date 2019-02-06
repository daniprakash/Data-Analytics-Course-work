
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
print(dataset)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

print(corpus)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 100)

#fitting and then transforming corpus into X(independent variable)
#all the other things need sparse matrix but dbscan needs dense matrix
Sparse_matrix = cv.fit_transform(corpus).todense()
print(Sparse_matrix)

#Density-Based Spatial Clustering of Applications with Noise.
from sklearn.cluster import DBSCAN
#eps = The maximum distance between two samples for them to be considered as in the same neighborhood.
#The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
clustering = DBSCAN(eps=0.3, min_samples=10).fit(X)
print(clustering.labels_)
clustering_labels_list = list()
clustering_labels_list = list(clustering.labels_)
print(clustering_labels_list)
print(type(clustering_labels_list))
print(len(dataset))


result_dbscan = []

for i in range(0, len(dataset)):
    result_par = []
    result_par.append(dataset['Review'][i])
    result_par.append(clustering_labels_list[i])
    result_dbscan.append(result_par)
    print(result_par)
  
print(result_dbscan) 


