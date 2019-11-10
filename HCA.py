# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 21:22:22 2019

@author: user
"""

#import necessary modules
import pandas as pd
df_sourcefile = pd.read_csv(r'C:\Users\user\Downloads\Attrition_data.csv')
print(df_sourcefile)

df_hr=df_sourcefile.copy()

df_hr.columns
df_hr.columns.to_series().groupby(df_hr.dtypes).groups

#--------data-----

#---------Count the number of missing in each column--------
print(df_hr.isnull().sum())
#--------end---------


#--------remove columns wit std dev=0-----------
df_hr=df_hr.drop(df_hr.std()[df_hr.std()== 0].index.values, axis=1)
#--------end-------
df_hr.shape

#-------categorical to numerical-

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le=LabelEncoder()

df_hr.head()
le_count=0
for col in df_hr.columns[1:]:
    if df_hr[col].dtype=='object':
        if len(list(df_hr[col].unique()))<=2:
            le.fit(df_hr[col])
            df_hr[col]=le.transform(df_hr[col])
            le_count+=1
           
print('{} columns were label encoded .' .format(le_count))
df_hr=pd.get_dummies(df_hr,drop_first=True)    


column_names = df_hr.columns
print(column_names)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,5))
Hr_col=list(df_hr.columns)
Hr_col.remove('Attrition')
Hr_col.remove('Employee_id')
for col in Hr_col:
    df_hr[col]=df_hr[col].astype(float)
    df_hr[[col]]=scaler.fit_transform(df_hr[[col]])
df_hr['Attrition']=pd.to_numeric(df_hr['Attrition'],downcast='float')
df_hr.head()
target=df_hr['Attrition'].copy()

df_hr.shape

#------------------------------end---------------

corr_matrix = df_hr.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

# Drop features
df_hr.drop(to_drop, axis=1, inplace=True)




#from sklearn.cluster import FeatureAgglomeration
#n_clusters = 6
#agglo = FeatureAgglomeration(n_clusters)
#agglo.fit(x_train)
#
#df_transformed = agglo.transform(x_train)
#df_transformed


#----Select more significant features-----
hr_vars=df_hr.columns.values.tolist()
y=['Attrition']
X=[i for i in hr_vars if i not in y]
df_hr.dtypes

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(df_hr[X],df_hr[y])
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=df_hr[X].columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
#-----End----

#------Split into test and train data----
cols=['Employee_id','Age', 'NumCompaniesWorked','IndustryExperience','TimeInTitle','JobSatisfaction','DistanceFromHome','WorkCultureSatisfaction','OverTime','TrainingTimesLastYear','WorkLifeBalance','NumCompaniesWorked']
X=df_hr[cols]

y=df_hr[['Employee_id','Attrition']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
y_train=y_train['Attrition']
#y_train=y_train['Attrition']


#====Fit the model----
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#----Validate the model on test data-----
y_pred=logreg.predict(X_test)

y_test['prediction']=y_pred

data=y_test.merge(X_test,on="Employee_id",how="left")
data.to_csv('data1.csv')
y_test=y_test['Attrition']


#-----Check Accuracy----
#X.dtypes
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')


#---Accuracyscore---
from sklearn.metrics import accuracy_score
print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(X_test))))


#-----Plot ROC curve=====

# calculate the fpr and tpr for all thresholds of the classification
y_pred=logreg.predict(X_test)
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
