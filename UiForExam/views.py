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
from sklearn.preprocessing import LabelEncoder

def welcome(request):
   
   return render(request, 'index.html')
# Create your views here.
def clean(request):
   df = pd.read_csv('data/malaria_clinical_data.csv')
  #  df.drop(columns=['SampleID','consent_given','location','Enrollment_Year','bednet','fever_symptom','Suspected_Organism','Suspected_infection','RDT','Blood_culture','Urine_culture','Taq_man_PCR','Microscopy','Laboratory_Results','rbc_count','RBC_dist_width_Percent',])
  #  y = df['Clinical_Diagnosis']
   temperatureMedian=df['temperature'].median(axis=0)
   df.fillna(temperatureMedian, inplace=True)
   parasite_densityMedian=df['parasite_density'].median(axis=0)
   df.fillna(parasite_densityMedian, inplace=True)
   wbc_countMedian=df['wbc_count'].median(axis=0)
   df.fillna(wbc_countMedian, inplace=True)

   hb_levelMedian=df['hb_level'].median(axis=0)
   df.fillna(hb_levelMedian, inplace=True)

   hematocritMedian=df['hematocrit'].median(axis=0)
   df.fillna(hematocritMedian, inplace=True)
   mean_cell_volumeMedian=df['mean_cell_volume'].median(axis=0)
   df.fillna(mean_cell_volumeMedian, inplace=True)

   mean_corp_hbMedian=df['mean_corp_hb'].median(axis=0)
   df.fillna(mean_corp_hbMedian, inplace=True)

   mean_cell_hb_concMedian=df['mean_cell_hb_conc'].median(axis=0)
   df.fillna(mean_cell_hb_concMedian, inplace=True)

   platelet_countMedian=df['platelet_count'].median(axis=0)
   df.fillna(platelet_countMedian, inplace=True)

   platelet_distr_widthMedian=df['platelet_distr_width'].median(axis=0)
   df.fillna(platelet_distr_widthMedian, inplace=True)

   mean_platelet_vlMedian=df['mean_platelet_vl'].median(axis=0)
   df.fillna(mean_platelet_vlMedian, inplace=True)

   neutrophils_percentMedian=df['neutrophils_percent'].median(axis=0)
   df.fillna(neutrophils_percentMedian, inplace=True)

   lymphocytes_percentMedian=df['lymphocytes_percent'].median(axis=0)
   df.fillna(lymphocytes_percentMedian, inplace=True)

   mixed_cells_percentMedian=df['mixed_cells_percent'].median(axis=0)
   df.fillna(mixed_cells_percentMedian, inplace=True)

   neutrophils_countMedian=df['neutrophils_count'].median(axis=0)
   df.fillna(neutrophils_countMedian, inplace=True)

   lymphocytes_countMedian=df['lymphocytes_count'].median(axis=0)
   df.fillna(lymphocytes_countMedian, inplace=True)

   lymphocytes_countMedian=df['lymphocytes_count'].median(axis=0)
   df.fillna(lymphocytes_countMedian, inplace=True)


   
   duplicates= df[df.duplicated()]
   df.drop_duplicates(inplace=True)

   result = df.to_string()
   return render(request, 'clean.html', {'result': result,'duplicates': duplicates,})

def accuracy(request):
    df = pd.read_csv('data/malaria_clinical_data.csv')
    df.drop(columns=['SampleID'], inplace=True)  # Drop the 'SampleID' column
    
    if 'Clinical_Diagnosis' not in df.columns:
        # Handle the absence of 'Clinical_Diagnosis' column
        # For example, you could choose a different target column or modify the dataset
        
        # Return an appropriate response or redirect to an error page
        return render(request, 'error.html', {'message': 'The column "Clinical_Diagnosis" is not present in the dataset.'})
    
    columns_to_drop = ['consent_given', 'location', 'Enrollment_Year', 'bednet', 'Suspected_Organism', 'Suspected_infection', 'RDT', 'Blood_culture', 'Urine_culture', 'Taq_man_PCR', 'Microscopy', 'Laboratory_Results', 'rbc_count', 'RBC_dist_width_Percent']
    
    x = df.drop(columns=columns_to_drop)
    
    # Encode the string labels in 'Clinical_Diagnosis' column
    label_encoder = LabelEncoder()
    x['Clinical_Diagnosis'] = label_encoder.fit_transform(x['Clinical_Diagnosis'].astype(str))
    
    y = x['Clinical_Diagnosis']
    
    # Fill missing values with medians
    median_values = df.median(axis=0)
    df.fillna(median_values, inplace=True)
    
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
    
    accuracy_RFC = accuracy_score(y_test, pred_RFC) * 100
    accuracy_SVC = accuracy_score(y_test, pred_SVC) * 100
    accuracy_LR = accuracy_score(y_test, pred_LR) * 100
    accuracy_DTC = accuracy_score(y_test, pred_DTC) * 100
    
    return render(request, 'accuracy.html', {'accuracy_RFC': accuracy_RFC, 'accuracy_SVC': accuracy_SVC, 'accuracy_LR': accuracy_LR, 'accuracy_DTC': accuracy_DTC})
def createJoblib(request):
# Split the dataset into features (X) and target variable (y)
  df = pd.read_csv('data/malaria_clinical_data.csv')
  x=df.drop(columns=['SampleID','consent_given','location','Enrollment_Year','bednet','Suspected_Organism','Suspected_infection','RDT','Blood_culture','Urine_culture','Taq_man_PCR','Microscopy','Laboratory_Results','rbc_count','RBC_dist_width_Percent',])
  y = df['linical_Diagnosis']
  genderMedian=df['linical_Diagnosis'].median()
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




   