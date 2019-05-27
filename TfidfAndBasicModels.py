import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator

############################################################################
# Visualization functions
############################################################################

def displayWordCloud(): # I called this function after cleaning the text
    spamtext = " ".join(text for text in data[data["spam"]==1].cleantext)
    hamtext=" ".join(text for text in data[data["spam"]==0].cleantext)
    wordcloud = WordCloud(max_words=100, background_color="white").generate(spamtext)
    wordcloud2= WordCloud(max_words=100, background_color="white").generate(hamtext)
# Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.imshow(wordcloud2, interpolation='bilinear')
    plt.axis("off")
    wordcloud.to_file("wordcloudspam.png")
    wordcloud2.to_file("wordcloudham.png")
    plt.show()


def DisplayConfusionMatrix(M,filename):
    
    plt.clf()
    plt.imshow(M, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Ham','Spam']
    plt.title('Ham or Spam Confusion Matrix - Valid Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(M[i][j]))
    plt.savefig(filename+'.png')
    plt.show()

def DisplayROC(classifier,filename):
    probs = classifier.predict_proba(X_valid)
    probs=probs[:,1]
    fpr, tpr, thresholds = roc_curve(y_valid, probs)
# plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
# show the plot
    plt.savefig('ROC Curve '+ filename+'.png')
    plt.show() 
    
############################################################################
# Data
############################################################################

df = pd.read_csv('spam.csv',encoding = "ISO-8859-1")
df.head()

data = df.iloc[:,:2]
data = data.rename(columns={"v1":"spam", "v2":"text"})

data.groupby("spam").count().plot.bar()
plt.show()

data['spam']= np.where(data['spam']=='spam',1,0)
data.head()

############################################################################
# Cleaning 
############################################################################

import re   
from string import punctuation
from nltk.stem import SnowballStemmer  
from nltk.corpus import stopwords      

def clean(text, stem_words=True):
        
    if type(text) != str or text=='':
        return ''

    # Clean the text (here i have 2-3 cases of pre-processing by sampling the data. You might need more)
    text = re.sub("\'s", " ", text) # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    
    # you might need more
    text = re.sub("\'m", " ", text)
    text = re.sub("\'re", " ", text)
    text = re.sub("\'ll", " ", text)
    text = re.sub("\'n't", " not ", text)
    
    # remove comma between numbers, i.e. 15,000 -> 15000
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
    
    # remove symbols (?!., etc.) - keep only words and numbers
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    #text = ''.join([sign for sign in text if sign not in punctuation])

    # if number, indicate a number
    text = re.sub('[0-9]+', 'number', text)
    
    # uncapitalize
    text = text.lower() 

    # split
    text = text.split()
    
    # stopwords
    stop = stopwords.words('english')
    text = [word for word in text if (word not in set(stop)) ]

    # stem
    if stem_words:
        snowball = SnowballStemmer("english")
        text = [snowball.stem(word) for word in text] # stemming
    
    # Return a list of words
    text = ' '.join(text) # the oppsite of splite (bring the words back together with a space btween them)
    return text

# With cleaning
#data['cleantext'] = data['text'].apply(clean)
# Without cleanning
data['cleantext'] = data['text']

#displayWordCloud()

############################################################################
# Features
############################################################################

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

y = data['spam']
X = data['cleantext']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state = 42)
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.3,random_state = 42)

# Count vectorizer
#vect = CountVectorizer()
#vect = CountVectorizer(max_features=1000)

# TF-IDF
vect = TfidfVectorizer(norm='l2', smooth_idf=True, use_idf=True)
#vect = TfidfVectorizer(norm='l2', smooth_idf=True, use_idf=True, max_features=1000)

# Fit and transform
vect.fit(X_train)  
X_train = vect.transform(X_train).toarray() 
X_valid = vect.transform(X_valid).toarray()
X_test = vect.transform(X_test).toarray()

# Feature names
new_names = vect.get_feature_names()
old_names = range(len(new_names))
X_train = pd.DataFrame(data=X_train)
X_train = X_train.rename(columns=dict(zip(old_names,new_names)))

############################################################################
# Model
############################################################################

#from sklearn.model_selection import RandomizedSearchCV

results = {}

# Logistic regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)
results['Logistic Regrassion'] = classifier

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
results['Naive Bayes'] = classifier

# Random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
results['Random Forest'] = classifier

############################################################################
# Evaluation
############################################################################

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve

# Baseline
prob = sum(y_train) / len(y_train)
y_baseline = np.random.choice([0,1], size=len(y_valid), p=[1-prob, prob])
print('accuracy',accuracy_score(y_valid,y_baseline))
print('roc_auc',roc_auc_score(y_valid, y_baseline))
print(classification_report(y_valid, y_baseline))

# Performance 
for k,v in results.items():
    # measures
    print('\n',k)
    predictions=v.predict(X_valid)
    print('accuracy',v.score(X_valid,y_valid))
    print('roc_auc',roc_auc_score(y_valid, predictions))
    print(classification_report(y_valid, predictions))
    # vizualization
    ConfusionMatrix = confusion_matrix(y_valid, predictions)
    DisplayConfusionMatrix(ConfusionMatrix, k)
    DisplayROC(v, k)

# Feature importance
impact = pd.DataFrame({ 'Feature':list(X_train), 'Importance':results['Random Forest'].feature_importances_ })
impact['Importance'] = impact['Importance'].round(decimals=5)
impact = impact.sort_values(by=['Importance'],ascending=False).reset_index(drop=True)
print(impact)

############################################################################
# Evaluation - test set
############################################################################

# Baseline
y_baseline = np.random.choice([0,1], size=len(y_test), p=[1-prob, prob])
print('accuracy',accuracy_score(y_test,y_baseline))
print('roc_auc',roc_auc_score(y_test, y_baseline))
print(classification_report(y_test, y_baseline))

# Performance 
for k,v in results.items():
    # measures
    print('\n',k)
    predictions=v.predict(X_test)
    print('accuracy',v.score(X_test,y_test))
    print('roc_auc',roc_auc_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    # vizualization
    ConfusionMatrix = confusion_matrix(y_test, predictions)
    DisplayConfusionMatrix(ConfusionMatrix, k)
    DisplayROC(v, k)