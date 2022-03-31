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


# In[17]:


for i in range (len (data)):
    if data.loc[i, "Gender"] == 'Male':
        data.loc[i, "Gender"] = 1
    else:
        data.loc[i, "Gender"] = 0
data.head(30)


# In[18]:


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


# In[19]:


fig, axes = plt.subplots(9, 1, figsize=(15, 100), sharey=True)
"""sns.histplot(data, ax=axes[0], x="Age", kde=True, color='r')
sns.histplot(data, ax=axes[1], x="Total Bilirubin", kde=True, color='c')
sns.histplot(data, ax=axes[2], x="Direct Bilirubin", kde=True)"""
lab = list(data.columns[2:])
print(lab)
color1 = ['r','g','b']
for i in range(len(lab)):
    sns.histplot(data, ax = axes[i], x = lab[i], kde = True, color = color1[i%3])


# In[20]:


#Fungsi untuk menghitung prior probability dengan masukan df(data point) dan Y(diagnosis)
def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    #Classes isinya dalah list dari df[Y] yang unik, yaitu 0 dan 1
    prior = [len(df[df[Y]==i])/len(df) for i in classes]
    #Disini perhitungan prior probability.
    return prior


# In[21]:


#Fungsi menghitung distribusi gaussian.
def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    #df akan berisi df[dataset dimana df[Y] bernilai 0 atau 1]
    #Jadi maksudnya, df akan berisikan semua data dimana df[Y] adalah 0 untuk iterasi pertama
    #Dan selanjutnya df akan berisikan semua data dimana df[Y] adalah 1 untuk interasi selanjutnya
    mean, std = df[feat_name].mean(), df[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std**2)) *  np.exp(-((feat_val-mean)**2 / (2 * std**2 )))
    #Fungsi perhitungan utama, rujuk rumus diatas
    return p_x_given_y


# In[22]:


def naive_bayes_gaussian(df, X, Y):
    # Features adalah semua column nomor 2 kekanan
    features = list(df.columns)[2:]

    # Panggil fungsi prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # Array kosong untuk menyimpan hasil perhtiugnan
    for x in X:
        # Kita iterasi untuk semua data sepanjang Xtest
        labels = sorted(list(df[Y].unique()))
        likelihood = [1 for i in range(len(labels))]
        #List comprehension
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])
        # calculate posterior probability (numerator only)
        post_prob = [likelihood[j] * prior[j] for j in range(len(labels))]
        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 


# In[23]:


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


# In[24]:


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
    #Seberapa akurat data kita
    recall = tp / (tp + fn)
    return 2 * (precision * recall )/ (precision + recall)


# In[26]:


train, test = TrainTestSplit(data, test_size= 0.2)

X_test = test.iloc[:,2:].values
Y_test = test.iloc[:,1].values

Y_pred = naive_bayes_gaussian(train, X=X_test, Y="Dataset")

cm = ConfusionMatrix(Y_test, Y_pred)
print("The confusion matrix is : \n{}".format(cm))
print("The accuracy is : {}" .format(f1(cm)))
print("The actual diagnosis is \n{} \nThe Predicted diagnosis based on the ML model is \n{}\nWhere 0 is benign and 1 is Malignant".format(Y_test, Y_pred))


# In[ ]:




