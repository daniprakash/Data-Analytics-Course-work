from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
documents = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
print(documents['Review'][999])
documents_list  = list()
for i in range(0, len(documents)):
    documents_list.append(documents["Review"][i])
print(documents_list)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents_list)

true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :100]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

result_kmeans = []
no_of_ones = 0
no_of_zeros = 0

y_pred = list()

for i in range(0, len(documents)):
    result_par = []
    result_par.append(documents['Review'][i])
    review_transform = vectorizer.transform([documents['Review'][i]])
    review_prediction = model.predict(review_transform)
    y_pred.append(review_prediction)
    print(review_prediction)
    if(review_prediction == 1):
        no_of_ones = no_of_ones + 1
    else:
        no_of_zeros = no_of_zeros + 1
    result_par.append(review_prediction)
    result_kmeans.append(result_par)
    print(result_par)

print(no_of_zeros)
print(no_of_ones)  
print(result_kmeans) 
print(y_pred)
y_actual = list()

for i in range(0 , len(documents)):
    y_actual.append(documents['Liked'][i])

print(len(y_actual))
print(len(documents))


from sklearn.metrics import confusion_matrix
cm_kmeans = confusion_matrix(y_actual,y_pred)
print(cm_kmeans)


Y = vectorizer.transform(["good place to be here"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["bad place to be here"])
prediction = model.predict(Y)
print(prediction)
 