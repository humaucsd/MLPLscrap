from __future__ import print_function
import math
import os.path
import random
import matplotlib.pyplot as plt


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import sklearn
import sklearn.datasets
import sklearn.ensemble
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular

np.random.seed(1)


import input_old

csvs2 = [f for f in os.listdir('ml2/data/sp14/op+type+size') if f.endswith('.csv')]
csvs = [f for f in os.listdir('ml2/data/fa15/op+type+size') if f.endswith('.csv')]

random.shuffle(csvs)
dfs = []
test = []
train = []

for csv in csvs:
	df, fs, ls = input_old.load_csv(os.path.join('ml2/data/fa15/op+type+size', csv), filter_no_labels=True, only_slice=False)

	if df is None:
		continue
	if df.shape[0] == 0:
		continue

	train.append(df)

for csv in csvs2:
	df2, fs2, ls2 = input_old.load_csv(os.path.join('ml2/data/sp14/op+type+size', csv), filter_no_labels=True, only_slice=False)

	if df2 is None:
		continue
	if df2.shape[0] == 0:
		continue

	test.append(df2)



train = pd.concat(train)
test = pd.concat(test)

classes = list(train.groupby(ls2))
#print(ls)
max_samples = max(len(c) for _, c in classes)
train = pd.concat(c.sample(max_samples, replace=True) for _, c in classes)




train_samps = train.loc[:,'F-InSlice':]
train_labels = train.loc[:,'L-DidChange']

# print test
test_samps = test.loc[:,'F-InSlice':]
test_labels = test.loc[:,'L-DidChange']
test_span = test.loc[:,'SourceSpan']

rf = RandomForestClassifier(n_estimators=30)
rf = rf.fit(train_samps.values, train_labels.values)

anses = rf.predict(test_samps.values)

# print (test_samps.index)
# print (test_samps.loc[0300.0])
temp = test_samps.loc[0300.0]
temp=temp.values

explainer = lime.lime_tabular.LimeTabularExplainer(train_samps.values, feature_names=fs, class_names=ls, discretize_continuous=True)
#i = np.random.randint(0, test_samps.values.shape[0])
# exp = explainer.explain_instance(temp[0], rf.predict_proba,  num_features=5)

# # exp.show_in_notebook(show_table=True, show_all=False)

# # fig = exp.as_pyplot_figure()

# exp.save_to_file('/tmp/oi.html')

k = 0
for t in temp:
	exp = explainer.explain_instance(t, rf.predict_proba,  num_features=10)
	exp.save_to_file('/tmp/' + str(k) + 'oi.html')
	k = k+1
	print (t)
	print (exp.as_list())