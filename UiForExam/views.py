from django.shortcuts import render
from django.http import HttpResponse
def welcome(request):
   return render(request, 'index.html')
# Create your views here.
def clean(request):
   return render(request, 'clean.html')
def accuracy(request):
    return render(request, 'accuracy.html')
def createJoblib(request):
    return render(request, 'createJoblib.html')
def result(request):
    return render(request, 'result.html')

   