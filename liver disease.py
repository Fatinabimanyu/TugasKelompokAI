#!/usr/bin/env python
# coding: utf-8

# # Liver Disease

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
sns.set_style("darkgrid")


data = pd.read_csv("Indian Liver Patient Dataset (ILPD).csv")

data['Gender'].replace({'Male': 1, 'Female': 0},inplace = True)
data['Dataset'].replace({1: 0, 2: 1},inplace = True)
data.head(30)


data["Dataset"].hist()
numOfYes = 0
numOfNo = 0
for i in range (len (data)):
    if data.loc[i, "Dataset"] == 1:
        numOfYes += 1
    else:
        numOfNo += 1
print("Yes = "+ str(numOfYes))
print("No = " + str(numOfNo))


fig, axes = plt.subplots(9, 1, figsize=(15, 100), sharey=True)
"""sns.histplot(data, ax=axes[0], x="Age", kde=True, color='r')
sns.histplot(data, ax=axes[1], x="Total Bilirubin", kde=True, color='c')
sns.histplot(data, ax=axes[2], x="Direct Bilirubin", kde=True)"""
lab = list(data.columns[2:])
print(lab)
color1 = ['r','g','b']
for i in range(len(lab)):
    sns.histplot(data, ax = axes[i], x = lab[i], kde = True, color = color1[i%3])


#Fungsi ini digunakan untuk menghitung prior probability dengan masukan df(data point) dan Y(diagnosis)
def calculate_priory(df, Y):
    classes = sorted(list(df[Y].unique()))
    #Classes isinya dalah list dari df[Y] yang unik, yaitu 0 dan 1
    prior = [len(df[df[Y]==i])/len(df) for i in classes]
    #Disini perhitungan prior probability.
    return prior


#Fungsi untuk menghitung distribusi gaussian.
def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    #df akan berisi df[dataset dimana df[Y] bernilai 0 atau 1
    #Jadi maksudnya disini, df disini akan berisikan semua data dimana df[Y] adalah 0 untuk iterasi pertama
    #Dan selanjutnya df akan berisikan semua data dimana df[Y] adalah 1 untuk interasi selanjutnya
    mean, std = df[feat_name].mean(), df[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std**2)) *  np.exp(-((feat_val-mean)**2 / (2 * std**2 )))
    #Fungsi perhitungan utama, rujuk rumus diatas
    return p_x_given_y


def naive_bayes_gaussian(df, X, Y):
    # Features semua column nomor 2 ke arah kanan
    features = list(df.columns)[2:]

    prior = calculate_priory(df, Y)

    Y_pred = []
    # Array kosong disini digunakan untuk menyimpan hasil dari perhtiugnan
    for x in X:
        # Kita iterasi untuk semua data sepanjang Xtest
        labels = sorted(list(df[Y].unique()))
        likelihood = [1 for i in range(len(labels))]
        #List of comprehension
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])
        # calculate the posterior probability (numerator only)
        post_prob = [likelihood[j] * prior[j] for j in range(len(labels))]
        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 


def TrainTestSplit(df, test_size):
    testSampleSize = len(df) * test_size
    if (testSampleSize - int(testSampleSize)) >= 0.5:
        testSampleSize =  int(testSampleSize) + 1
    else:
        testSampleSize = int(testSampleSize)
    indices = []
    trainArr = []
    testArr = []
    count = 0
    for i in range(testSampleSize):
        y = random.randint(0, len(df))
        while y in indices:
            y = random.randint(0, len(df))
        indices.append(y)
        count += 1
    
    for i in indices:
        testArr.append(df.iloc[i:i+1])
    for i in range(len(df)):
        if i not in indices:
            trainArr.append(df.iloc[i:i+1])
    return pd.concat(trainArr).sort_index(ascending = True), pd.concat(testArr).sort_index(ascending = True)


def ConfusionMatrix (yTest, yPred):
    matrix = np.array([[0,0], [0,0]])
    for i in range (len(yTest)):
        if yPred[i] == 1:
            if yTest[i] == 1:
                matrix[0][0] += 1
            else:
                matrix [0][1] += 1
        else:
            if yTest[i] == 1:
                matrix[1][0] += 1
            else:
                matrix [1][1] += 1
    return matrix


def f1 (confusionMatrix):
    tp = confusionMatrix [0][0]
    fp = confusionMatrix[0][1]
    fn = confusionMatrix [1][0]

    precision = tp / (tp + fp)
    #Seberapa akurat data yang kita miliki
    recall = tp / (tp + fn)
    return 2 * (precision * recall )/ (precision + recall)


train, test = TrainTestSplit(data, test_size= 0.2)

X_test = test.iloc[:,2:].values
Y_test = test.iloc[:,1].values

Y_pred = naive_bayes_gaussian(train, X=X_test, Y="Dataset")

cm = ConfusionMatrix(Y_test, Y_pred)
print("The confusion matrix is : \n{}".format(cm))
print("The accuracy is : {}" .format(f1(cm)))
print("The actual diagnosis is \n{} \nThe Predicted diagnosis based on the ML model is \n{}\nWhere 0 is benign and 1 is Malignant".format(Y_test, Y_pred))

#membandingkan dengan logistic regression yang menggunakan sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix 

X = data[['Gender', 'Dataset', 'Age', 'Total Bilirubin', 'Direct Bilirubin',
       'Alkhpos', 'SGPT', 'SGOT', 'Total Proteins', 'Albumin', 'A/G Ratio']]
y = data.Dataset

#membagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(np.nan_to_num(X), y, test_size=0.2)
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

logreg = LogisticRegression(solver='lbfgs', max_iter=1000)

logreg.fit(X_train, y_train)

# memprediksi keluaran
log_predicted= logreg.predict(X_test)
logreg_score = round(logreg.score(X_train, y_train) * 100, 2)
logreg_score_test = round(logreg.score(X_test, y_test) * 100, 2)

# print output
print('Logistic Regression Training Score: n', logreg_score)
print('Logistic Regression Test Score: n', logreg_score_test)
print('Accuracy: n', accuracy_score(y_test,log_predicted))
print('Confusion Matrix: n', confusion_matrix(y_test,log_predicted))
print('Classification Report: n', classification_report(y_test,log_predicted))
