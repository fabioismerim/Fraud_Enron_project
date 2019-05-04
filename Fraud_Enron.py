#!/usr/bin/env python
# coding: utf-8

# # Projeto:  Identificar fraudes nos e-mails da Enron

# ## Resumo do Projeto

# Em 2000, Enron era uma das maiores empresas dos Estados Unidos. Já em 2002, ela colapsou e quebrou devido a uma fraude que envolveu grande parte da corporação. Resultando em uma investigação federal, muitos dados que são normalmente confidenciais, se tornaram públicos, incluindo dezenas de milhares de e-mails e detalhes financeiros para os executivos dos mais altos níveis da empresa. Neste projeto, você irá bancar o detetive, e colocar suas habilidades na construção de um modelo preditivo que visará determinar se um funcionário é ou não um funcionário de interesse (POI). Um funcionário de interesse é um funcionário que participou do escândalo da empresa Enron. Para te auxiliar neste trabalho de detetive, nós combinamos os dados financeiros e sobre e-mails dos funcionários investigados neste caso de fraude, o que significa que eles foram indiciados, fecharam acordos com o governo, ou testemunharam em troca de imunidade no processo.

# ## Objetivo do Projeto

# O Objetivo deste projeto é utilizar o machine learning para identificar um POI (pessoa de interesse) baseado nos dados financeiros e email dos funcionários da Enron.
# 
# Esse projeto é dividio em quatro partes:
# 1. <b> Explorar o Dataset</b>
#     Data Cleaning, Análise e remoção de Outlier
# 2. <b> Processamento de features</b>
#     Envolve criação de features, feature scallig, feature selection e feature transform
# 3. <b> Escolha do Algoritmo </b>
#     Escolher três algoritmos para testar 
# 4. <b> Validação </b>
#     Ver qual dos algoritmos se sai melhor
#     
# É importante ressaltar também, que o projeto foi realizado na versão <b> Python 3 </b>
#     

# In[1]:


# importando bibliotecas

import sys
import pickle
sys.path.append("../tools/")

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.grid_search import GridSearchCV
from time import time

import pandas as pd
from matplotlib import pyplot as plt

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


# ### 1.Explorando Dataset

# In[2]:


#carregando os dados e transformando em um DataFrame

df_enron = pd.DataFrame.from_records(list(data_dict.values()))

#ajustando índice para nome dos funcionários:
employees = pd.Series(list(data_dict.keys()))
df_enron.set_index(employees, inplace=True)
df_enron.head()


# In[3]:


df_enron.shape


# In[4]:


df_enron.info()


# In[5]:


df_enron.isnull().sum().sum()


# Os tipos de dados estão todos em string. vamos converter em float

# In[6]:


df_enron_new = df_enron.apply(lambda x : pd.to_numeric(x, errors = 'coerce')).copy().fillna(0)
df_enron_new.head()


# In[7]:


df_enron_new.dtypes


# In[8]:


# Não vamos utilizar a coluna e-mail em nossas análises, portanto vamos excluí-las
df_enron_new.drop('email_address', axis = 1, inplace = True)

# Checking the changed shape of df
df_enron_new.shape


# In[9]:


#quantidade de POI's e não-POI's.
poi_count = df_enron.groupby('poi').size()
print ("Número de POI'S : ", poi_count.iloc[1])
print ("Número de não-POI's : ",poi_count.iloc[0])


# In[10]:


# número de features
len(df_enron_new.columns)


# ### Investigação de Outliers

# #### Financial Features: Bonus e Salário

# In[11]:


#vamos plotar um gráfico de dispersão

plt.scatter(df_enron_new['salary'][df_enron_new['poi'] == True],df_enron_new['bonus'][df_enron_new['poi'] == True], color = 'r',
           label = 'POI')
plt.scatter(df_enron_new['salary'][df_enron_new['poi'] == False],df_enron_new['bonus'][df_enron_new['poi'] == False],color = 'b',
           label = 'Not-POI')
    
plt.xlabel("Salario")
plt.ylabel("Bonus")
plt.title("Gráfico de dispersão do salário vs bônus")
plt.legend(loc='upper left')
plt.show()


# No gráfico acima podemos observar um alto valor de salário e bônus, indicando um posível outlier. Vamos checar.

# In[12]:


#checando o mais alto valor de salário de um não POI
(df_enron_new['salary'][df_enron_new['poi'] == False]).argmax()


# In[13]:


#chcando o mais alto valor de bônus de um não POI
(df_enron_new['bonus'][df_enron_new['poi'] == False]).argmax()


# #### Removendo Outlier

# In[14]:


#deletando a linha 'total' do dataset
df_enron_new.drop('TOTAL', axis = 0, inplace = True)

#Plotando gráfico de dispersão novamente

plt.scatter(df_enron_new['salary'][df_enron_new['poi'] == True],df_enron_new['bonus'][df_enron_new['poi'] == True], color = 'r',
           label = 'POI')
plt.scatter(df_enron_new['salary'][df_enron_new['poi'] == False],df_enron_new['bonus'][df_enron_new['poi'] == False],color = 'b',
           label = 'Not-POI')
    
plt.xlabel("Salario")
plt.ylabel("Bonus")
plt.title("Gráfico de dispersão do salário vs bônus")
plt.legend(loc='upper left')
plt.show()


# Após a remoção do outlier, os dados ficaram mais fáceis de compreender. Podemos observar que os valores de salário e bônus são maiores para os POI's do que os não-POI's.
# Outro ponto interessante é que somente dois POI's possuem a combinação salário e bônus elevados.

# #### Investigando dados de e-mails

# Vamos checar a quantidade de e-mails enviados entre POI's e não-POI's. É comum esperarmos uma quantidade elevada de transações de e-mails entre os POI's

# In[15]:


plt.scatter(df_enron_new['from_poi_to_this_person'][df_enron_new['poi'] == False],
            df_enron_new['from_this_person_to_poi'][df_enron_new['poi'] == False],
            color = 'b', label = 'Not-POI')

plt.scatter(df_enron_new['from_poi_to_this_person'][df_enron_new['poi'] == True],
            df_enron_new['from_this_person_to_poi'][df_enron_new['poi'] == True],
            color = 'r', label = 'POI')

    
plt.xlabel('from_poi_to_this_person')
plt.ylabel('from_this_person_to_poi')
plt.title("Gráfico de dispersão da quantidade de e-mails de origem e destino entre POI e a pessoa")
plt.legend(loc='upper right')
plt.show()


# In[16]:


(df_enron_new['from_poi_to_this_person'][df_enron_new['poi'] == False]).argmax()


# In[17]:


df_enron_new[df_enron_new.from_poi_to_this_person>500]


# Criando novas features: proporção de e-mails enviados de/para POI's

# In[18]:


df_enron_new['fraction_mail_from_poi'] = round(df_enron_new['from_poi_to_this_person'].fillna(0.0)/df_enron_new['from_messages'].fillna(0.0),2)
df_enron_new['fraction_mail_to_poi'] = round(df_enron_new['from_this_person_to_poi'].fillna(0.0)/df_enron_new['to_messages'].fillna(0.0),2)


# In[19]:


#checando se há NaN values
df_enron_new.isnull().sum().sum()


# após a criação das colunas, temos novamente alguns NaN's em nosso dataframe. Precisamos removê-los

# In[20]:


#removendo NaN values

df_enron_new[['fraction_mail_from_poi', 'fraction_mail_to_poi']] = df_enron_new[['fraction_mail_from_poi', 'fraction_mail_to_poi']].fillna(value=0)
df_enron_new.sample(5)


# In[21]:


df_enron_new.isnull().sum().sum()


# In[22]:


df_enron_new.info()


# In[23]:


#convertendo para um dicionário e armazenando
data_dict = df_enron_new.to_dict(orient='index')
my_dataset = data_dict


# In[24]:


my_dataset


# ### Features

# Temos as seguintes features em nosso dataset
# 
# - 17 financial features : ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (todos em dólares americanos (USD))
# 
# - 6 Email features : ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (as unidades aqui são geralmente em número de emails; a exceção notável aqui é o atributo ‘email_address’, que é uma string)
# 
# - POI: [‘poi’] (atributo objetivo lógico (booleano), representado como um inteiro)
# 
# Vou inserir também as features que criamos ['fraction_mail_from_poi, 'fraction_mail_to_poi']

# Vamos treinar nossos classificadores com a lista completa de features e depois vamos ver quais são as melhores features que devemos selecionar

# In[25]:


features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
                 'long_term_incentive', 'restricted_stock', 'director_fees', 'fraction_mail_from_poi', 'fraction_mail_to_poi',
                 'to_messages', 'from_poi_to_this_person', 'from_messages','from_this_person_to_poi', 'shared_receipt_with_poi']


# In[26]:


#extraindo features e labels do dataset:
from feature_format import featureFormat, targetFeatureSplit
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[27]:


print(data)


# In[28]:


labels


# ### Seleção de Algoritmos

# In[29]:


# importando bibliotecas para validação

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import scikitplot as skplt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import cross_validation


# In[30]:


#dividindo os dados em treino e teste
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)


# In[31]:


# Stratified ShuffleSplit cross-validator
X = np.array(features)
y = np.array(labels)


# Utilizarei 4 cassificadores neste projeto
# - Naive Bayes
# - Decision Tree
# - K-means
# - SVM

# In[32]:


#importando os classificadores
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from tester import dump_classifier_and_data
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# ### Performance dos algoritmos: Precision, Accuracy e Recall

# In[33]:


# Classificador 1: Gaussian Naive Bayes

sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.33,random_state = 42)
sss.get_n_splits(X,y)
#print(sss)

for train_index, test_index in sss.split(X,y):
    features_train, features_test = X[train_index], X[test_index]
    labels_train, labels_test= y[train_index], y[test_index]
    

clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print ("tempo de treinamento:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print ("tempo de previsão:", round(time()-t0, 3), "s")

#Acurácia
accuracy = accuracy_score(labels_test,pred)
#Precisão
prec = precision_score(labels_test,pred)
#Recall
recall_s = recall_score(labels_test,pred)
#f1-score
f1 = f1_score(labels_test, pred)

print("Acurácia do classificador GaussianNB é: ", accuracy)
print("A precisão do classificador GaussianNB é: ", prec)
print("O recall do classificador GaussianNB é: ", recall_s)
print("O f-1 score do classificador GaussianNB é: ", f1)


# In[34]:


# Classificador 2: Decision Tree
sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.33,random_state = 42)
sss.get_n_splits(X,y)
#print(sss)

for train_index, test_index in sss.split(X,y):
    features_train, features_test = X[train_index], X[test_index]
    labels_train, labels_test= y[train_index], y[test_index]
    
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf = clf.fit(features_train, labels_train)

print ("tempo de treinamento:", round(time()-t0, 3), "s")
t0 = time()

pred = clf.predict(features_test)
print ("tempo de previsão:", round(time()-t0, 3), "s")

#Acurácia
accuracy = accuracy_score(labels_test,pred)
#Precisão
prec = precision_score(labels_test, pred)
#Recall
recall_s = recall_score(labels_test, pred)
#f1-score
f1 = f1_score(labels_test,pred)

print("Acurácia do classificador DecisionTree é: ", accuracy)
print("A precisão do classificador DecisionTree é: ", prec)
print("O recall do classificador DecisionTree é: ", recall_s)
print("O f-1 score do classificador DecisionTree é: ", f1)


# In[35]:


#Classificador 3: K-means
sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.33,random_state = 42)
sss.get_n_splits(X,y)
#print(sss)

for train_index, test_index in sss.split(X,y):
    features_train, features_test = X[train_index], X[test_index]
    labels_train, labels_test= y[train_index], y[test_index]
    
clf = KMeans(n_clusters=2)
clf = clf.fit(features_train, labels_train)
print ("tempo de treinamento:", round(time()-t0, 3), "s")
t0 = time()

pred = clf.predict(features_test)
print ("tempo de previsão:", round(time()-t0, 3), "s")

#Acurácia
accuracy = accuracy_score(labels_test,pred)
#Precisão
prec = precision_score(labels_test,pred)
#Recall
recall_s = recall_score(labels_test,pred)
#f1-score
f1 = f1_score(labels_test,pred)

print("Acurácia do classificador K-means é: ", accuracy)
print("A precisão do classificador K-means é: ", prec)
print("O recall do classificador K-means é: ", recall_s)
print("O f-1 score do classificador K-means é: ", f1)


# In[36]:


# Classificador 4: SVM
sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.33,random_state = 42)
sss.get_n_splits(X,y)
#print(sss)

for train_index, test_index in sss.split(X,y):
    features_train, features_test = X[train_index], X[test_index]
    labels_train, labels_test= y[train_index], y[test_index]
    
clf = SVC(C = 10000, kernel = "rbf")

t0 = time()
clf.fit(features_train, labels_train)
print ("tempo de treinamento:", round(time()-t0, 3), "s")

#Previsão
t0 = time()
pred = clf.predict(features_test)
print ("tempo de previsão:", round(time()-t0, 3), "s")


#Acurácia
accuracy = accuracy_score(labels_test,pred)
#Precisão
prec = precision_score(labels_test,pred)
#Recall
recall_s = recall_score(labels_test,pred)
#f1-score
f1 = f1_score(labels_test,pred)

print("Acurácia do classificador SVM é: ", accuracy)
print("A precisão do classificador SVM é: ", prec)
print("O recall do classificador SVM é: ", recall_s)
print("O f-1 score do classificador SVM é: ", f1)


# In[ ]:




