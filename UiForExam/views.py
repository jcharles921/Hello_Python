from django.shortcuts import render
# from django.http import HttpResponse
from django.conf import settings
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def welcome(request):
   
   return render(request, 'index.html')
# Create your views here.
def clean(request):
   df = pd.read_csv('data/QUIZ4L2.csv')
   df.drop(columns=['genre'])
   y = df['genre']
   genderMedian=df['gender'].median()
   df.fillna(genderMedian, inplace=True)
   duplicates= df[df.duplicated()]
   df.drop_duplicates(inplace=True)

   result = df.to_string()
   return render(request, 'clean.html', {'result': result,'duplicates': duplicates,})

def accuracy(request):
       df = pd.read_csv('data/QUIZ4L2.csv')
       x=df.drop(columns=['genre'])
       y = df['genre']
       genderMedian=df['gender'].median()
       df.fillna(genderMedian, inplace=True)
       df.drop_duplicates(inplace=True)
       x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
       model_RFC = RandomForestClassifier()
       model_SVC = SVC()
       model_LR = LogisticRegression()
       model_DTC = DecisionTreeClassifier()

       model_RFC.fit(x_train, y_train)
       model_SVC.fit(x_train, y_train)
       model_LR.fit(x_train, y_train)
       model_DTC.fit(x_train, y_train)

       pred_RFC = model_RFC.predict(x_test)
       pred_SVC = model_SVC.predict(x_test)
       pred_LR = model_LR.predict(x_test)   
       pred_DTC = model_DTC.predict(x_test)

       accuracy_RFC = accuracy_score(y_test, pred_RFC)*100
       accuracy_SVC = accuracy_score(y_test, pred_SVC)*100
       accuracy_LR = accuracy_score(y_test, pred_LR)*100
       accuracy_DTC = accuracy_score(y_test, pred_DTC)*100
       return render(request, 'accuracy.html', {'accuracy_RFC': accuracy_RFC, 'accuracy_SVC': accuracy_SVC, 'accuracy_LR': accuracy_LR, 'accuracy_DTC': accuracy_DTC,})
       
def createJoblib(request):
# Split the dataset into features (X) and target variable (y)
  df = pd.read_csv('data/QUIZ4L2.csv')
  x=df.drop(columns=['genre'])
  y = df['genre']
  genderMedian=df['gender'].median()
  df.fillna(genderMedian, inplace=True)
  df.drop_duplicates(inplace=True)

# # Create and train the Decision Tree Classifier
  model = DecisionTreeClassifier()
  model.fit(x, y)
  model_path = os.path.join(settings.BASE_DIR, 'models', 'model.joblib')
  joblib.dump(model, model_path)
  return render(request, 'createJoblib.html')
def result(request):
    return render(request, 'result.html')
def predictform(request):
    model_path = os.path.join(settings.BASE_DIR, 'models', 'model.joblib')
    model = joblib.load(model_path)

    if request.method == 'POST':
        age = request.POST.get('age')
        sex = request.POST.get('sex')
        prediction = model.predict([[age, sex]])[0]
        return render(request, 'formresult.html', {'prediction': prediction})




   
