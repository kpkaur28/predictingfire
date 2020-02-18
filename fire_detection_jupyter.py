import pandas as pd 
import numpy as np 
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neural_network import MLPClassifier 
from sklearn.decomposition import PCA 
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc, roc_auc_score, precision_score, recall_score 
from sklearn.metrics import classification_report 

#names = ['X', 'Y', 'temp', 'wind', 'rain', 'fire']
train=pd.read_csv("forestfires.csv", header=0)
test=pd.read_csv("decision_tree.csv", header=0)
#print(train)
#print(test)

print('Train Data Shape: {}'.format(train.shape))
print('Test Data Shape: {}'.format(test.shape))

#Checking missing values
train_missing_values=train.isnull().sum()
#print("Train missing values:", train_missing_values)

#Data Preprocessing
#Clean the missing values in both training and testing data 
train_data=train.dropna(axis=0,how="any")
test_data=test.dropna(axis=0,how="any")
print('Train Clean Data Shape: {}'.format(train_data.shape))
print('Test Clean Data Shape: {}'.format(test_data.shape))

#Finding the number of fires
train_data["fire"].value_counts()
print(train_data["fire"].value_counts())

sns.countplot(x=train_data["fire"])
#plt.title("No. of fires", fontsize=15)
#plt.show()

train_data["temp"].value_counts()
train_data.groupby(["temp"])["fire"].value_counts()

sns.countplot(x=train_data["temp"],hue=train_data["fire"])
#plt.title("temp vs fire",fontsize=15)
#plt.show()

str_data=train_data.select_dtypes(include=['object'])
str_dt=test_data.select_dtypes(include=['object'])

int_data=train_data.select_dtypes(include=['integer',"float"])
int_dt=test_data.select_dtypes(include=['integer',"float"])

#Defining features and label
xtrain=train_data[["temp", "wind"]]     #using only temperature and wind
#print(xtrain)
ytrain=train_data["fire"]
#print(ytrain)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(xtrain,ytrain, test_size=0.25)
#print(x_train)

#--------------------Building NaiveBayes Model
print("\n--------------Building NaiveBayes Model\n")
model = GaussianNB()
model.fit(x_train,y_train)
#print(model.fit(x_train,y_train))

predict=model.predict(x_test)

df = pd.read_csv("naive_bayes.csv", header=0)
predictions = []
for index, row in df.iterrows():
    pred1 = model.predict([row[["temp", "wind"]]])
    #print("pred1 is:", pred1)
    predictions.append(pred1)
predictions = list(map(int, predictions))
df['pred_naive_bayes'] = predictions
print(df)
df.to_csv('naive_bayes_output.csv')

print("\n")

##Look into how scoring works
test_score=model.score(x_test,y_test)
print("NBtest_score:", test_score)

train_score=model.score(x_train,y_train)
print("NBtrain_score:", train_score)

#Cross Validation
from sklearn.model_selection import cross_validate
cv_results = cross_validate(model,xtrain,ytrain,cv=5)
#print(cv_results)

#NaiveBayes Confusion Matrix
nb_conf_mtr=pd.crosstab(y_test,predict)
#print(nb_conf_mtr)

#Classification Report for NaiveBayes
nbreport=classification_report(y_test,predict)
#print(nbreport)

#------------------- Buidling Decision Tree Model 
print("\n--------------Building Decison Tree Model\n")
dt_mod=DecisionTreeClassifier(criterion='entropy',max_depth=8)
dt_mod.fit(x_train,y_train)

y_pred=dt_mod.predict(x_test)

##Open the test data excel and then input the prediction values as a separate column
df = pd.read_csv("decision_tree.csv", header=0)
predictions = []
for index, row in df.iterrows():
    pred1 = dt_mod.predict([row[["temp", "wind"]]])
    #print("pred1 is:", pred1)
    predictions.append(pred1)
predictions = list(map(int, predictions))
df['pred_decision_tree'] = predictions
print(df)
df.to_csv('decision_tree_output.csv')

print("\n")

ts_dt_score=dt_mod.score(x_test,y_test)
print("DTest_score:",ts_dt_score)

x_pred=dt_mod.predict(x_train)
ts_dt_score=dt_mod.score(x_train,y_train)
print("DTrain_score:",ts_dt_score)

dt_report=classification_report(y_test,y_pred)
#print(dt_report)

#--------------------Building Neural Network
print("\n--------------Building Neural Network\n")

mlp_model=MLPClassifier(max_iter=1000)
mlp_model.fit(x_train,y_train)
#print(mlp_model.fit(x_train,y_train))

mlp_predict=mlp_model.predict(x_test)
#print(mlp_predict)

df = pd.read_csv("neural_network.csv", header=0)
predictions = []
for index, row in df.iterrows():
    pred1 = mlp_model.predict([row[["temp", "wind"]]])
    #print("pred1 is:", pred1)
    predictions.append(pred1)
predictions = list(map(int, predictions))
df['pred_neural_network'] = predictions
print(df)
df.to_csv('neural_network_output.csv')

print("\n")

ts_mlp_score=mlp_model.score(x_test,y_test)
print("NNtest_score: ", ts_mlp_score)

tr_mlp_score=mlp_model.score(x_train,y_train)
print("NNtrain_score: ", tr_mlp_score)

nn_cv_results=cross_validate(mlp_model,xtrain,ytrain)
#print(nn_cv_results)

"""
#-------------------Dimensionality Reduction PCA - reduces the dimension of data, process faster and makes the score better
print("\n-------------Dimensionality Reduction PCA\n")

pca=PCA(n_components=3)
pca=pca.fit(xtrain)
#print(pca)

PCtrain=pca.transform(x_train)
#print(PCtrain)

PCtest=pca.transform(x_test)
#print(PCtest)

print("Applying PCA to all the models, and then predicting score")

print("\n--------------PCA on NaiveBayes Model\n")
model = GaussianNB()
model.fit(PCtrain,y_train)
predict=model.predict(PCtest)

pca_nb_score=model.score(PCtest,y_test)
print("PCA_NBtest_score:", pca_nb_score)

pca_train_score=model.score(PCtrain,y_train)
print("PCA_NBtrain_score:", pca_train_score)

print("\n--------------PCA on Decison Tree Model\n")
pca_dt_mod=DecisionTreeClassifier(criterion='entropy',max_depth=8)
pca_dt_mod.fit(PCtrain,y_train)

y_pred=pca_dt_mod.predict(PCtest)
pca_dt_ts_score=pca_dt_mod.score(PCtest,y_test)
print("DTest_score:",pca_dt_ts_score)

x_pred=pca_dt_mod.predict(PCtrain)
pca_dt_tr_score=pca_dt_mod.score(PCtrain,y_train)
print("DTrain_score:",pca_dt_tr_score)

print("\n--------------PCA on Neural Network\n")
pca_mlp_model=MLPClassifier(hidden_layer_sizes=(80,),alpha=0.0001)
pca_mlp_model.fit(PCtrain,y_train)

pca_ts_mlp_score=pca_mlp_model.score(PCtest,y_test)
print("NNtest_score: ", pca_ts_mlp_score)

pca_tr_mlp_score=pca_mlp_model.score(PCtrain,y_train)
print("NNtrain_score: ", pca_tr_mlp_score)

print("\n")
score=pd.DataFrame({"score": {"nb_score": test_score, 
                              "dt_score": ts_dt_score, 
                              "nn_score": ts_mlp_score}})
print(score)
print("\n")

score1=pd.DataFrame({"pca_score":{"pca_nb_score": pca_nb_score,
                                  "pca_dt_score": pca_dt_ts_score, 
                                  "pca_nn_score": pca_ts_mlp_score}})
print(score1)
"""