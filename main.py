from concurrent.futures.thread import _worker
import json
from multiprocessing import Value
from tempfile import tempdir
from xmlrpc.client import boolean
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from numpy import array
import pandas as pd
from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report,  confusion_matrix
import time
from sklearn.model_selection import GridSearchCV

# following function was used to tunning the parameters of SVM machine 
def finding_best_params(X_train,y_train,X_test,y_test):
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid','linear']}
    grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2,cv=2)
    grid.fit(X_train,y_train)
    print(grid.best_estimator_)
    grid_predictions = grid.predict(X_test)
    print(confusion_matrix(y_test,grid_predictions))
    print(classification_report(y_test,grid_predictions))

def tfidf():

    start_time = time.time()
    pathTest =  "./data/essay-corpus.json"
    pathsplit = "./data/train-test-split.csv"

    fileObject = open(pathTest, "r", encoding='utf-8')
    jsonContent = fileObject.read()

    Data_dic = json.loads(jsonContent)
    df_spliter = pd.read_csv(pathsplit, sep=';', header=None)
    
    
    # for splitting the data
   
    split_dict = {}
    train_data = []
    test_data = []
    
    i = 0
    for item in df_spliter[1]:
       split_dict[i] = item
       i = i+1

    for item in Data_dic:
       id =  item['id']
       if split_dict[id] == 'TRAIN':
            train_data.append(item)
       else:
            test_data.append(item)


    Traindata=pd.DataFrame(train_data) 
    Testdata=pd.DataFrame(test_data)



    # Data Pre-processing  
    Saperators = ['?', ',','.','!',';',':']

    Traindata = Testdata[~Testdata['text'].isin(Saperators) ]
    Testdata = Traindata[~Traindata['text'].isin(Saperators) ]

    #remove blank space
    Testdata['text'].dropna(inplace=True)
    Traindata['text'].dropna(inplace=True)

    #converting float to string
    Testdata['text'] = [str(entry) for entry in Testdata['text']]
    Traindata['text'] = [str(entry) for entry in Traindata['text']]

    #Lowercase the text
    Testdata['text'] = [entry.lower() for entry in Testdata['text']]
    Traindata['text'] = [entry.lower() for entry in Traindata['text']]

    
           
    #tf idf
    Train_X = Traindata['text']
    Train_Y = Traindata['confirmation_bias']

    Test_X = Testdata['text']
    Test_Y = Testdata['confirmation_bias']

    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    #Vectorization of corpus using TF-IDF, Features extraction
    Tfidf_vec = TfidfVectorizer()
    Tfidf_vec.fit(Train_X)

    Train_X_Tfidf = Tfidf_vec.transform(Train_X)
    Test_X_Tfidf = Tfidf_vec.transform(Test_X)


    # parameter tunning in the following function

    finding_best_params(Train_X_Tfidf,Train_Y,Test_X_Tfidf,Test_Y)

    SVM = svm.SVC(C=1, gamma=1, kernel='sigmoid')
    SVM.fit(Train_X_Tfidf, Train_Y)
    predictions = SVM.predict(Test_X_Tfidf)
    # print(predictions)
    # print(classification_report(Test_Y,predictions,labels=np.unique(predictions)))

    predictions_decoded= (Encoder.inverse_transform(predictions)).tolist()
    id_list = Traindata['id'].tolist()
    
    predict_csv = []
    
    for i in range(0,len(predictions_decoded)):
        temp={}
        temp['id']= str(id_list[i])
        temp['confirmation_bias']=predictions_decoded[i]
        predict_csv.append(temp)

    prediction_json = json.dumps(predict_csv)
    with open('predictions.json', 'w') as f:
        f.write(prediction_json)
        
    #timer to check the execution time
    print("Process finished --- %s seconds ---" % (time.time() - start_time))

    # # #for simpling pre-process the text using gensim library
    # #for test
    # test_text= testdata.text.apply(gensim.utils.simple_preprocess)
    # test_cfbias= str(testdata.confirmation_bias.apply(gensim.utils.simple_tokenize))
    
    # model_test_text= gensim.models.Word2Vec(test_text, min_count = 1, vector_size = 1000, window = 5, sg = 1)
    # model_test_text.build_vocab(test_text, progress_per=1000)
    # model_test_text.train(test_text, total_examples= model_test_text.corpus_count, epochs=model_test_text.epochs)
    
    # model_test_cfbias= gensim.models.Word2Vec(test_cfbias, min_count = 1, vector_size = 1000, window = 5, sg = 1)
    # model_test_cfbias.build_vocab(test_cfbias, progress_per=1000)
    # model_test_cfbias.train(test_cfbias, total_examples= model_test_text.corpus_count, epochs=model_test_text.epochs)

    # #for train
    # train_text= traindata.text.apply(gensim.utils.simple_preprocess)
    # train_cfbias= str(traindata.confirmation_bias.apply(gensim.utils.simple_tokenize))

    # model_train_text= gensim.models.Word2Vec(train_text, min_count = 1, vector_size = 1000, window = 5, sg = 1)
    # model_train_text.build_vocab(train_text, progress_per=1000)
    # model_train_text.train(train_text, total_examples= model_test_text.corpus_count, epochs=model_test_text.epochs)
    
    # model_train_cfbias= gensim.models.Word2Vec(train_cfbias, min_count = 1, vector_size = 1000, window = 5, sg = 1)
    # model_train_cfbias.build_vocab(train_cfbias, progress_per=1000)
    # model_train_cfbias.train(train_cfbias, total_examples= model_test_text.corpus_count, epochs=model_test_text.epochs)

def main():
    ## YOUR CODE HERE
    print("it works!")
    tfidf()
    pass



main()