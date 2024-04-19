import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


dataset = pd.read_csv("hiring.csv")
print(dataset)

dataset['experience'].fillna(0,inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

x = dataset.iloc[:,:3]
print(x.head)

def convert_to_integer(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

x['experience'] = x['experience'].apply(lambda x : convert_to_integer(x))
#print(x.head())

y = dataset.iloc[:,-1]
print(y.head)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
