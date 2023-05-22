import nltk
from django.shortcuts import render
from django.http import JsonResponse
from artSum.classes.user import User
from artSum.classes.article import Article
from artSum.classes.naiveBayes import NaiveBayes
from artSum.classes.neuralNetwork import NeuralNetwork
from artSum.classes.decisionTree import DecisionTree
nltk.download('punkt')
nltk.download('stopwords')

user = User()
nb = NaiveBayes()
nn = NeuralNetwork()
dt = DecisionTree()
# Create your views here.

def home(request):
    return render(request, 'home.html', {'name': 'Django'})

def upload(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        pdf_file = request.FILES.get('pdf_file', None)
        text = user.upload_file(pdf_file)
        article = Article(pdf_file.name,text)
        num_words = len(text.split())
        return JsonResponse({'text': text, 'num_words': num_words})
    
    else:
        return render(request, 'home.html')

# Generate summaries
def summarize(request):
    # Get text from POST request
    text = request.POST.get('text', '')
    mode = request.POST.get('mode', 'naive_bayes') # Default to naive bayes if mode is not provided
    summary = user.summarize(text , mode)
    summary_count = len(summary.split())
    return JsonResponse({'summary': summary, 'summary_count': summary_count})