# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:41:56 2022

@author: pratiksanghvi
"""
#--------------------------------------------------------------
import pandas as pd
import pickle
#--- Read the dataset------------------------------------------
message = pd.read_csv("D:\\Learn & Projects\\Data Science Extra\\Coding\\NLP\\Spam-Ham\\SMSSpamCollection",sep='\t',names= ['label','message'])



#Data cleaning / Data preprocessing
import re
import nltk
nltk.download('stopwords')


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wL = WordNetLemmatizer()
corpus = []
for i in range(len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message['message'][i])
    review = review.lower()
    review = review.split()
#    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
# Using lemmetization instead of stemming to check if accuracy is above 98%
    review = [wL.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Bag of Words for converting to document frequnecy
# from sklearn.feature_extraction.text import CountVectorizer
    
# cv = CountVectorizer(max_features= 5000)
# X = cv.fit_transform(corpus).toarray()    
  
# Using TF-IDF to see the accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()

#Create a pickle file for transform
filename = 'transformTheMessageToVector.pkl'
pickle.dump(cv,open(filename,'wb'))

  
#Create dummy variable for the dependent output deature
y= pd.get_dummies(message['label'])
# 1 coloumn is enough for representing spam=1/ham=0 message
y = y.iloc[:,1].values 


#---Train test split--------------------------
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)  
   
# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
nbInitialize = MultinomialNB()
spam_detect_model = nbInitialize.fit(X_train, y_train)
naive_score = nbInitialize.score(X_test,y_test)


y_pred=spam_detect_model.predict(X_test)

# Confusion matrix
from sklearn.metrics import confusion_matrix
conf_m = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)
print(acc)

#BOW & Stem , Acc = 98%
# TFIDF , Lemm , Acc = 97.21%
    
# Create a pickle file
filename = 'nlp.model.pkl'
pickle.dump(nbInitialize,open(filename,'wb'))
    
    
   
    
    
    
    
    
    
    
    
    
    
