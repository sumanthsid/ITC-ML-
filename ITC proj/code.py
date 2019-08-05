


#kfelrkfbk.bsg;
import pandas as pd
import numpy as np
dataset = pd.read_csv("datasettwoo.csv")


X=dataset.iloc[:,0:248].values
y=dataset.iloc[:,248].values

#missingvalues
from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X[:,:])
X=imp.transform(X[:,:])


#feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)


#feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=15)
clf = clf.fit(X, y)
clfli=list(clf.feature_importances_)
clflisort = clfli.copy()
clflisort.sort(reverse=True)

clfdict=dict(enumerate(clfli))
clfdictsort=clfdict.copy()
clfdictsort=sorted(clfdict, key=clfdict.get, reverse=True)

#clfdictsort=sorted(clfdictsort.items(), key=lambda x: x[1],reverse=True)

'''
clfin=[]
for i in clflisort:
    clfin.append(clflisort.index(i))


'''
'''index=[]

l=list(clf.feature_importances_)
for i in range(40):
    i=l.index(max(l))
    index.append(i)
    l.remove(max(l))
    '''



model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new.shape


'''
ind=[]   
l=list(clf.feature_importances_)
l.sort(reverse=True)
for i in range(len(l)):
    ind.append(l[i])
l.sort(reverse=True)
'''

index=[164,185,0,96,190,78,81,94,220,67,97,157,71,84,193,85,191,168,95,65,64,167,186,178,230,169,170,82,224,232,162,246,244,21,20,25,\
       226,156,80,14,236,83,136,1,19,12,189,87,214,129,22,29,212,112,15,176,117,153,26,155,242,79,239,171,73,45,138,184,72,231,114,13,\
       188,59,158,223,93,179,115,172,109,123,89,88,163,209,28,177,192,46,69,108,18,222,58,92,235,44,183,70,119,243,221,98,234,195,86,2,110,122,194,47,175,205,31,30,62,165,204,127,247,24,197,27,139,130,33,6,113,161,126,198,107,120,63,116,42,111,118,225,137,237,200,201,227,99,66,219,4,131,206,196,43,128,182,208,203,38,187,240,154,32,241,68,121,61,229,180,10,41,217,211,5,215,7,216,199,233,202,35,11,16,34,245,181,3,8,57,147,49,124,100,174,106,159,135,101,207,51,125,166,152,23,142,149,213,143,210,9,37,148,160,145,144,40,17,90,173,48,60,50,133,146,132,141,56,36,151,134,150,53,55,140,52,39,54,74,75,76,77,91,102,103,104,105,218,228,238]

#index=sorted(range(len(clfli)), key=lambda i: clfli[i])[-2:]       
#index.reverse()    
    
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X[:,index[:40]], y, test_size=0.3, random_state=0)
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
 

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=33, max_depth=None, min_samples_split=2, random_state=0)
clf = clf.fit(X_train, y_train)
y_pred2=clf.predict(X_train)
y_pred=clf.predict(X_test)
'''

#classifier svmlinear

from sklearn.svm import SVC
classifier1 = SVC(gamma='auto',kernel='linear', C=500)
classifier1.fit(X_train,y_train)
y_pred2=classifier1.predict(X_train)
y_pred=classifier1.predict(X_test)

#classifier svmpoly
from sklearn.svm import SVC
classifier2 = SVC(gamma='auto', C=999,kernel='poly')
classifier2.fit(X_train,y_train)
#y_pred2=classifier2.predict(X_train)
y_pred=classifier2.predict(X_test).

#classifier svm rbf
from sklearn.svm import SVC
classifier3 = SVC(gamma='auto', C=999,kernel='rbf')
#classifier3.fit(X_train,y_train)
#y_pred2=classifier3.predict(X_train)
#y_pred=classifier3.predict(X_test)

#MAJORITYVOTING
from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[('linear', classifier1), ('poly', classifier2), ('rbf', classifier3)], voting='hard')
eclf1 = eclf1.fit(X_train, y_train)
y_pred2=eclf1.predict(X_train)
y_pred=eclf1.predict(X_test)
'''
from sklearn.metrics import confusion_matrix
confusionmatrix=confusion_matrix(y_test,y_pred)
'''
count=0

#Displayng accuracy
for i in range(0,len(y_test)):
    if y_pred[i]==y_test[i]:
        count=count+1
print("Testing Accuracy :",(count/len(X_test)*100))

count2=0
for i in range(0,len(y_train)):
    if y_pred2[i]==y_train[i]:
        count2=count2+1
print("Training Accuracy :",(count2/len(X_train)*100))
    
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
plt.plot(range(248),clfli)
plt.show()