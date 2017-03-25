import math
import os.path
import random
random.seed()

from sklearn import tree
import numpy as np
import pandas as pd

import pydotplus
from IPython.display import Image  

import input

from operator import add

import matplotlib.pyplot as plt


import plotly.plotly as py
import plotly.graph_objs as go

csvs = [f for f in os.listdir('ml2/data/fa15/op+type+size') if f.endswith('.csv')]
random.shuffle(csvs)
dfs = []
test = []
train = []
numcsv = 0
for csv in csvs:
	numcsv = numcsv+1
	df, fs, ls = input.load_csv(os.path.join('ml2/data/fa15/op+type+size', csv), filter_no_labels=True, only_slice=False)

	if df is None:
		continue
	if df.shape[0] == 0:
		continue

	dfs.append(df)


	if numcsv < 250:
		test.append(df)
	else:
		train.append(df)


# print (len(dfs))   
df = pd.concat(dfs)
train = pd.concat(train)
test = pd.concat(test)

# print (len(test))  
# print (len(train))  


# print (df.shape)
# print df
classes = list(train.groupby(ls))
#print(ls)
max_samples = max(len(c) for _, c in classes)
train = pd.concat(c.sample(max_samples, replace=True) for _, c in classes)
# print (len(train))  
#print df.shape
#print type(df)
#list_keys = [ k for k in df ]
#print list_keys
# print samps

#print sum(df['L-DidChange'].values)
# print df['L-DidChange'].index



train_samps = train.loc[:,'F-InSlice':]
train_labels = train.loc[:,'L-DidChange']

# print test
test_samps = test.loc[:,'F-InSlice':]
test_labels = test.loc[:,'L-DidChange']

# print test.iloc[1]
# print test.values[1]


# dflist = []
# keylist = []
# for key, value in df.iteritems():
#     temp = value
#     tempk = key
#     dflist.append(temp)
#     keylist.append(tempk)
# Y = dflist[0]
# X = dflist[2:]


clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_samps.values, train_labels.values)
# print test_samps
# print test_samps.values
anses = clf.predict(test_samps.values)
resacc = anses + 2*test_labels.values

#-------------------------------------
#Expression size
# y = test_samps.loc[:,'F-Expr-Size'].values
# pairs = zip(resacc,y)
# x1 = list(filter(lambda x : x>0,  (map(lambda x: x[1] if x[0] == 0 else 0, pairs))))
# x2 = list(filter(lambda x : x>0,  (map(lambda x: x[1] if x[0] == 1 else 0, pairs))))
# x3 = list(filter(lambda x : x>0,  (map(lambda x: x[1] if x[0] == 2 else 0, pairs))))
# x4 = list(filter(lambda x : x>0,  (map(lambda x: x[1] if x[0] == 3 else 0, pairs))))



# trace1 = go.Histogram(
#     x=x1,
#     name='True Negative',
#     marker=dict(
#         color='r',
#     ),
#     opacity=1
# )

# trace2 = go.Histogram(
#     x=x2,
#     name='False Positive',
#     marker=dict(
#         color='b',
#     ),
#     opacity=0.75
# #    nbinx
# )

# trace3 = go.Histogram(
#     x=x3,
#     name='False Negative',
#     marker=dict(
#         color='r',
#     ),
#     opacity=0.75
# )

# trace4 = go.Histogram(
#     x=x4,
#     name='True Positive',
#     marker=dict(
#         color='b',
#     ),
#     opacity=1
# )



# data = [trace1, trace2, trace3, trace4]

# layout = go.Layout(
#     title='Sampled Results',
#     xaxis=dict(
#         title='Expression Size'
#     ),
#     yaxis=dict(
#         title='Count'
#     ),
#     bargap=0.2,
#     bargroupgap=0.1
# )
# fig = go.Figure(data=data, layout=layout)
# py.iplot(fig, filename='Expression Size Analysis')


#-----PLOTTING
# dot_data = tree.export_graphviz(clf, out_file=None,
# 	feature_names=fs, 
# 	filled=True, 
# 	rounded=True) 
# graph = pydotplus.graph_from_dot_data(dot_data) 
# graph.write_png('fa1504.png')
#---------

# graph.render(filename='img/g1')
# graph.write_pdf("smallfa15.pdf") 
#--------------


# print anses
# print test_labels.values
# print sum(anses)/len(anses)
# print sum(test_labels.values)/len(test_labels.values)


#testanses =test_labels.values
resacc = anses + 2*test_labels.values
#acc = 1-((sum(abs(anses - test_labels.values)))/3600)

#lol = test_labels.add((-1)*anses)

#print lol
#-------importances

# imps = clf.feature_importances_
# imp_features = [(y,x) for (y,x) in sorted(zip(imps,fs))]
# imp_features.reverse()
# for elem in imp_features:
#         print elem  
#------------------

#print map(lambda x : clf.predict_proba(x), test_samps.values)
prob_score = clf.predict_proba(test_samps.values)
prob_error = [item[1] for item in prob_score]

# print prob_error

ll = zip(prob_error, resacc, test_samps.values)

score = pd.DataFrame(data=ll, index=test_labels.index, columns=['Error Probability','B', 'vects'])
# print score

print 'recall is ' + str(sum(anses * test_labels.values)/sum(test_labels.values))
print 'precision is ' + str(sum(anses * test_labels.values)/sum(anses))

yay1 = 0
yay2 = 0
yay3 = 0
tots = 0
tp = 0

heat = [0] * len(test_samps.values[0])

for labelind in list(set(test_labels.index)):
	#print labelind
	temp = score.loc[labelind]
	temp = temp.values
	# print labelind
	if len(temp) < 3:
		continue
	tots = tots+1
	topn = temp[np.argsort(temp[:,0])]
	# print topn
	# print 'lol'
	# print topn[-3:]
	a3 = 0
	a2 = 0
	a1 = 0
	if (abs(topn[-3][1] -3) <= 0.1) :
		a3 = 1
		tp = tp+1
		trunced =  topn[-3][2]#[1 if x > 1 else x for x in topn[-3][2]]
		heat = map(add, heat, trunced)
	if (abs(topn[-2][1] -3) <= 0.1) :
		a3 = 1
		a2 = 1
		tp = tp+1
		trunced =  topn[-2][2]#[1 if x > 1 else x for x in topn[-2][2]]
		heat = map(add, heat, trunced)
	if (abs(topn[-1][1] -3) <= 0.1) :
		a3 = 1
		a2 = 1
		a1 = 1
		tp = tp+1
		trunced =  topn[-1][2] #[1 if x > 1 else x for x in topn[-1][2]]
		heat = map(add, heat, trunced)		
	yay1 = yay1+a1	
	yay2 = yay2+a2
	yay3 = yay3+a3

print "precision for top 3"
print 'top 1' 
print float(yay1)/tots
print 'top 2' 
print float(yay2)/tots
print 'top 3'
print float(yay3)/tots
# print tots
# print tp
# print sum(test_labels.values)
print "recall for top 3"
print tp/sum(test_labels.values)
# print heat
# print fs

# data = [go.Bar(
#             x=fs,
#             y=heat
#     )]

#py.iplot(data, filename='fa15new-bar')