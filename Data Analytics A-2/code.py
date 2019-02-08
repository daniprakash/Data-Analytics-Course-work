# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:44:53 2018

@author: JEPRSDD
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

app_rev = pd.read_csv('App_Reviews_50.csv')
apple_rev = pd.read_csv('Apple_Reviews_50.csv')
ford_rev = pd.read_csv('Ford_Reviews_50.csv')
movie_rev = pd.read_csv('Movie_Reviews_50.csv')
res_rev = pd.read_csv('Restaurant_Reviews_50.csv')


plt.suptitle('Bar')

i = 0
for df,name in zip([app_rev, apple_rev, ford_rev, movie_rev, res_rev], ['Apps','Apple','Ford','Movie','Restaurant']):
	i = i +1
	plt.subplot(1, 5, i)
	ax = sns.barplot(x=df['id'],y=df['percentage'])
	ax.set(xlabel=name)
	ax.set_xticklabels([])
plt.show()


plt.suptitle('Scatter plot')

i = 0
for df,name in zip([app_rev, apple_rev, ford_rev, movie_rev, res_rev], ['Apps','Apple','Ford','Movie','Restaurant']):
	i = i +1
	plt.subplot(1, 5, i)
	ax = sns.scatterplot(x=df['id'],y=df['percentage'])
	ax.set(xlabel=name)
	ax.set_xticklabels([])
plt.show()



plt.suptitle('Histogram')

i = 0
for df,name in zip([app_rev, apple_rev, ford_rev, movie_rev, res_rev], ['Apps','Apple','Ford','Movie','Restaurant']):
	i = i +1
	plt.subplot(1, 5, i)
	ax = sns.distplot(df['percentage'])
	ax.set(xlabel=name)
	ax.set_xticklabels([])
plt.show()


plt.suptitle('Violin plot')

i = 0
for df,name in zip([app_rev, apple_rev, ford_rev, movie_rev, res_rev], ['Apps','Apple','Ford','Movie','Restaurant']):
	i = i +1
	plt.subplot(1, 5, i)
	ax = sns.violinplot(df['percentage'])
	ax.set(xlabel=name)
	ax.set_xticklabels([])
plt.show()


plt.suptitle('Box Whisker plot')

i = 0
for df,name in zip([app_rev, apple_rev, ford_rev, movie_rev, res_rev], ['Apps','Apple','Ford','Movie','Restaurant']):
	i = i +1
	plt.subplot(1, 5, i)
	ax = sns.boxplot(df['percentage'])
	ax.set(xlabel=name)
	ax.set_xticklabels([])
plt.show()


