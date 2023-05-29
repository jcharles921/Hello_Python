from django.urls import path
from . import views

urlpatterns = [
    path('', views.welcome, name='welcome'),
    path('clean/', views.clean, name='clean'),
    path('accuracy/', views.accuracy, name='accuracy'),
    path('createJoblib/', views.createJoblib, name='createJoblib'),
    path('result/', views.result, name='result'),
    path('result/predict', views.predictform, name='predict'),
]
