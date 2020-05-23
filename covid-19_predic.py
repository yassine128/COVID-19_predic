"""
Author: Yassine Seddaoui
Gmail: yassineseddaoui@gmail.com
Phone: 514-238-7019
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("This program was made by Yassine Seddaoui aka Cursedbuddy ©")
#PRENDRE LE FICHIER
coronavirus = pd.read_csv('covid_data.csv')
#coronavirus.head()


#TRAITER LE FICHIER ET SUPPRIMER LES (DATA) INUTILES
#coronavirus['death'] = coronavirus['death'].astype('Int64')
#coronavirus['age'] = coronavirus['age'].astype('Int64')
coronavirus = coronavirus[['gender', 'age', 'death']]
coronavirus = coronavirus[coronavirus['gender'].notna()]
coronavirus = coronavirus[coronavirus['age'].notna()]
coronavirus['gender'].replace(['male', 'female'], [0, 1], inplace=True)
#coronavirus.tail()

#REMPLACER TOUTES LES MAUVAISES (DATA) PAR 1
coronavirus['death'].replace(['2/21/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/21/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/19/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/19/2020'], [1], inplace=True)
coronavirus['death'].replace(['02/01/20'], [1], inplace=True)
coronavirus['death'].replace(['2/27/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/25/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/22/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/24/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/23/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/26/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/23/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/23/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/23/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/25/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/27/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/26/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/28/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/13/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/26/2020'], [1], inplace=True)
coronavirus['death'].replace(['2/14/2020'], [1], inplace=True)

#Besoin de (sex), (age), (death)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
y = coronavirus['death']
y=y.astype('int')
X = coronavirus.drop('death', axis=1)

#ENTRAINER LE MODÈLE
model.fit(X, y)
model.score(X, y)

def survie(model):
    gender = int(input('What is your sex? [male/female][0/1]: '))
    age = int(input('How old are you?: '))
    x = np.array([gender, age]).reshape(1, 2)
    predic = model.predict(x)
    test = model.predict_proba(x).T
    survive = test.item(0)
    die = test.item(1)
    #print(survive)
    #print(die)
    survive_text = str(survive*100)
    die_text = str(die*100)
    #print(model.predict(x))    
    #print(model.predict_proba(x))
    if predic == 0:
        print("You have " + survive_text + " % chance of surviving and " + die_text + " % chance of dying")
    if predic == 1:
        print("You have " + survive_text + " % chance of surviving and " + die_text + " % chance of dying")

survie(model)
