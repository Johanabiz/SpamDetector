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
    probs = classifier.predict(X_validation)
    probs = probs > 0.5
    probs=probs[:,1]
    fpr, tpr, thresholds = roc_curve(y_validation, probs)
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
    
    # Empty question
    
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

data['cleantext'] = data['text'].apply(clean)
#displayWordCloud()

############################################################################
# Split
############################################################################

from sklearn.model_selection import train_test_split

y = data['spam'].values
X = data['cleantext']
X_train_df,X_test_df,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state = 42)
X_train_df,X_validation_df,y_train,y_validation = train_test_split(X_train_df,y_train,test_size=0.3,random_state = 42)

############################################################################
# Features (tokenization)
############################################################################

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Fit
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_df)
# Convert
X_train = tokenizer.texts_to_sequences(X_train_df)
X_validation = tokenizer.texts_to_sequences(X_validation_df)
X_test = tokenizer.texts_to_sequences(X_test_df)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1
# Pad sequences with zeros
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_validation = pad_sequences(X_validation, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

############################################################################
# trained embeddings - word2wec
############################################################################

# WORD2VEC

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import Word2Vec  

# data
token_sent = []
for el in X_train_df:
    token_sent.append(el.split())

# model
embedding_dim = 50
w2v_model=Word2Vec(token_sent,size=embedding_dim,window=3,seed=42,workers=4)
w2v_model.train(token_sent,total_examples=len(token_sent),epochs=10)

# saving the model
#w2v_model.save('w2v_model.pickle')
#w2v_model = Word2Vec.load('w2v_model.pickle')

# vocabulary 
vocab=list(w2v_model.wv.vocab)
print('Vocabulary length: ', len(vocab))
#print(vocab)

# vector of a particular model. note that it is 100 dimensional as specified.
print(w2v_model.wv.get_vector('free'))
# most similar words to a given word
print()
print(w2v_model.wv.most_similar('free',topn=10))

W2V_matrix = np.zeros((vocab_size, embedding_dim))
for i,word in enumerate(vocab):
    vector = w2v_model.wv[word]
    W2V_matrix[i] = np.array(vector, dtype=np.float32)[:embedding_dim]
    
############################################################################
# Model
############################################################################

results = {}

from keras.models import Sequential
from keras import layers

# Neural network
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, weights=[W2V_matrix], input_length=maxlen,trainable=True))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=3, verbose=True, validation_data=(X_validation, y_validation), batch_size=10)

results['word2vec'] = model 

############################################################################
# Pre-trained
############################################################################

# PRE-TRAINED EMBEDDINGS

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#This is a function that reads a pre-trained embeddings file and returns a matrix embeddings for the dataset we are working with.
#Inputs are the filepath, the size of the embeddings (should match the pre-trained ones) and the word_indices as created by a tokenizer on our data
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

# Load
embedding_matrix = create_embedding_matrix('glove.6B.50d.txt',tokenizer.word_index, embedding_dim)
print(embedding_matrix.shape)

# Sparsity
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print(nonzero_elements / vocab_size)

############################################################################
# Model 
############################################################################

# Neural network
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen,trainable=True))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=5, verbose=True, validation_data=(X_validation, y_validation), batch_size=10)

results['pre trained'] = model 

############################################################################
# Evaluation
############################################################################

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve

# Performance 
for k,v in results.items():
    # measures
    print('\n',k)
    predictions=v.predict(X_validation)
    predictions = predictions > 0.5
    print("Accuracy: %0.2f%%" % (100 * accuracy_score(y_validation, predictions)))  
    print('roc_auc',roc_auc_score(y_validation, predictions))
    print(classification_report(y_validation, predictions))
    # vizualization
    ConfusionMatrix = confusion_matrix(y_validation, predictions)
    DisplayConfusionMatrix(ConfusionMatrix, k)
   # DisplayROC(v, k) try to do better

"""
print('\nTrain')
y_pred = model.predict(X_train)
y_pred = y_pred > 0.5
print("Accuracy: %0.2f%%" % (100 * accuracy_score(y_train, y_pred)))  
print(classification_report(y_train, y_pred))

print('\nValidation')
y_pred = model.predict(X_validation)
y_pred = y_pred > 0.5
print("Accuracy: %0.2f%%" % (100 * accuracy_score(y_validation, y_pred)))  
print(classification_report(y_validation, y_pred))


"""
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve

# Performance 
for k,v in results.items():
    # measures
    print('\n',k)
    predictions=v.predict(X_test)
    predictions = predictions > 0.5
    print("Accuracy: %0.2f%%" % (100 * accuracy_score(y_test, predictions)))  
    print('roc_auc',roc_auc_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    # vizualization
    ConfusionMatrix = confusion_matrix(y_test, predictions)
    DisplayConfusionMatrix(ConfusionMatrix, k)
   # DisplayROC(v, k) try to do better





























