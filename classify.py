import json
import pandas as pd
import re
import nltk
import pickle
import csv
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

nltk.download('stopwords')

#opens test dataset or prediction dataset
with open('test_file.json') as json_file:
    ds = pd.read_json(json_file, orient = 'records')
    test_dataset = ds.iloc[:, ~ds.columns.isin(['date','text'])]
    training_words = [];
	
	#adds words from training set
    with open('words.txt', 'r') as wordList:
        for items in wordList:
            training_words.append(items.strip())

    #strips every word in the training words and adds it into the dataframe for prediction
    for i in training_words:
        i = i.strip()
        if i != ''  and i != 'stars' and i !='useful' and i!= 'cool' and i!= 'funny' and i not in set(stopwords.words('english')):
            test_dataset[i] = 0
    test_dataset['stars1'] = 0
    test_dataset['useful1'] = 0
    test_dataset['cool1'] = 0
    test_dataset['funny1'] = 0

    #Goes through the test/prediction data and fills the dataframe if words appear
    for nums, i in enumerate(ds['text'].loc[:]):
        num = nums
        words = i.split()
        words = [re.sub('[^a-zA-Z]', "", c).lower() for c in words]
        for i in words:
            if i != '' and i in training_words and i != 'stars' and i != 'useful' and i != 'cool' and i != 'funny':
                test_dataset.at[num, i] += 1
            if i == 'stars':
                test_dataset.at[num, 'stars1'] += 1
            if i == 'useful':
                test_dataset.at[num, 'useful1'] += 1
            if i == 'cool':
                test_dataset.at[num, 'cool1'] += 1
            if i == 'funny':
                test_dataset.at[num, 'funny1'] += 1
    #opens the classifier 
    with open('clasifier.pickle', 'rb') as handle:
        classifier = pickle.load(handle)

    x_test = test_dataset.loc[:,test_dataset.columns != 'stars'].values
    y_pred = classifier.predict(x_test)
    #writes it to the prediction file
    with open('predictions.csv', 'w', newline = '') as prediction:
        writer = csv.writer(prediction)
        writer.writerow(['prediction'])
        for i in y_pred:
            writer.writerow([i])
