import math
import os.path
import random
import csv
import sys
random.seed()
import matplotlib.pyplot as plt

from sklearn import tree
import numpy as np
import pandas as pd

import input

csvs = [f for f in os.listdir('ml2-master2/data/fa15/op+context-count+type+size') if f.endswith('.csv')]
random.shuffle(csvs)
dfs = []
for mcsv in csvs:
	f = open('ml2-master/data/op+context-count+type+size/' + mcsv)
	reader = csv.reader(f)
	skip = 1
	for row in reader:
		if skip == 1:
			skip = 0
			heads = row
			continue
		dfs.append(row)
	f.close()

random.shuffle(dfs)
validate = dfs[0:2500]
Yv = [item[0] for item in validate]
Xv = [item[2:] for item in validate]

train = dfs[2501:]

Y = [item[0] for item in train]
X = [item[2:] for item in train]
print len(X)
print len(Y)
print Y[1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
anses = clf.predict(Xv[0:1000])
print(anses)
acc = 0
for i in range (0,1000):
	if anses[i] == Yv[i]:
		acc = acc+1
print (acc)

importances = clf.feature_importances_
#print(heads)

#plt.plot(heads[2:], importances)
lol = heads[2:]
#b = [importances for (lol,importances) in sorted(zip(lol,importances))]
b = [lol for (importances,lol) in sorted(zip(importances,lol))]
#print b