# Importing the libraries
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
X = cv.fit_transform(corpus).toarray()
print(X)


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(X)

print(cluster.labels_)  
print(len(cluster.labels_))
cluster_labels_list = list(cluster.labels_)
print(cluster_labels_list)  


result_hier = []

for i in range(0, len(dataset)):
    result_par = []
    result_par.append(dataset['Review'][i])
    result_par.append(cluster_labels_list[i])
    result_hier.append(result_par)
    print(result_par)
  
print(result_hier) 


y_actual = list()

for i in range(0 , len(dataset)):
    y_actual.append(dataset['Liked'][i])

print(len(y_actual))


from sklearn.metrics import confusion_matrix
cm_hier = confusion_matrix(y_actual,cluster_labels_list)
print(cm_hier)


plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')  

