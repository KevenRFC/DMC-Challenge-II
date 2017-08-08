#Importar Librarias
import pandas as pd
import numpy as np

# Cargar Datos
train_ori = pd.read_excel("E:/MODELOS_DS/COMPETENCIAS NACIONALES/DMC CHALLENGE/train_universidad.xlsx")
test_ori = pd.read_excel("E:/MODELOS_DS/COMPETENCIAS NACIONALES/DMC CHALLENGE/test_universidad.xlsx")

stop_words = pd.read_csv("E:/MODELOS_DS/COMPETENCIAS NACIONALES/DMC CHALLENGE/stop_words_4.txt",
                         encoding = 'latin-1')

stop_words = np.array(stop_words)

train_ori.columns = ['COD_ENCUESTADO',
                     'NOMBRE_CAMPUS',
                     'NIVEL_ACTUAL',
                     'CLAVE_CARRERA',
                     'CICLO',
                     'COMENTARIO',
                     'IND_GEA',
                     'IND_DELEGADO',
                     'CANT_CURSOS',
                     'IND_DEPORT_CALIF',
                     'NPS']

test_ori.columns = ['COD_ENCUESTADO',
                     'NOMBRE_CAMPUS',
                     'NIVEL_ACTUAL',
                     'CLAVE_CARRERA',
                     'CICLO',
                     'COMENTARIO',
                     'IND_GEA',
                     'IND_DELEGADO',
                     'CANT_CURSOS',
                     'IND_DEPORT_CALIF']

# Revisión Inicial
train_ori.info()
test_ori.info()

# Feature Engineering
train = pd.DataFrame(train_ori[['COMENTARIO','NPS']].values, columns = ['COMENTARIO','NPS'])
test = pd.DataFrame(test_ori[['COMENTARIO']].values, columns = ['COMENTARIO'])
test["NPS"] = np.nan

dataset = pd.concat([train,test], axis = 0)
dataset.reset_index(drop = True , inplace = True)

# -------------------------------------------------------------------------------------------

# Eliminar tildes
import unicodedata
def eliminar_tildes(cadena):
    return ''.join((c for c in unicodedata.normalize('NFD',cadena) if unicodedata.category(c) != 'Mn'))

# Cleaning the texts
import re
from nltk.stem.snowball import SnowballStemmer

len_dataset = len(dataset)
corpus = []

for i in range(0, len_dataset):
    review = dataset['COMENTARIO'][i]
    review = eliminar_tildes(review)
    review = re.sub('[^a-zA-Zñ]', ' ', review)
    review = review.lower()
    review = review.split()
    stemmer = SnowballStemmer('spanish')
    review = [(word) for word in review if not word in stop_words]
    review = [stemmer.stem(word) for word in review ]
    review = ' '.join(review)
    corpus.append(review)


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 378) 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[0:20000, 1].values.astype("int")


head_X = cv.vocabulary_
head_X = pd.concat([pd.DataFrame(list(head_X.keys())),pd.DataFrame(list(head_X.values()))], axis = 1)
head_X.columns = ['palabra','orden']
head_X.sort_values(by = ['palabra'] , ascending = True , inplace = True)

X_train = X[0:20000]
y_train = y

####################################################################################################
# Entrenar el modelo: MLPClassifier
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(random_state = 1234 , max_iter = 50 , learning_rate_init=0.05,
                           alpha = 0.1)
classifier = classifier.fit(X_train, np.array(y_train).ravel())

# Predecir sobre el Training Dataset
y_pred_proba = pd.DataFrame(classifier.predict_proba(X_train), columns = ["Proba_1","Proba_2","Proba_3","Proba_4"])
y_pred = classifier.predict(X_train)

# Evaluacion del Modelo: 
    ## Matriz de Confusión - TRAINING
from sklearn import metrics as mt
cm = mt.confusion_matrix(y_train, y_pred)
print(cm)
accuracy = mt.accuracy_score(y_train,y_pred)
print ("Accuracy : ", round(accuracy,3)) # Accuracy : 0.607  - 0.693

    # log_loss  - TRAINING
from sklearn.metrics import log_loss
log_loss_pred = log_loss(y_train,y_pred_proba)
print ("log loss : ", round(log_loss_pred,3)) # log loss : abc - 0.77

      
# Cross Validation
from sklearn import cross_validation
from sklearn.metrics import log_loss
cv = cross_validation.KFold(len(X_train), n_folds=10, random_state = 0)

results_train = []
results_test = []
# "Error_function" can be replaced by the error function of your analysis
for traincv, testcv in cv:
        probas_train = classifier.fit(X_train[traincv], y_train[traincv]).predict_proba(X_train[traincv])
        probas_test = classifier.fit(X_train[traincv], y_train[traincv]).predict_proba(X_train[testcv])
        log_loss_pred_train = log_loss(y_train[traincv],probas_train)
        log_loss_pred_test = log_loss(y_train[testcv],probas_test)
        results_train.append(log_loss_pred_train)
        results_test.append(log_loss_pred_test)
        print("Ok: " , str(log_loss_pred_train), " - ", str(log_loss_pred_test))

result_final_train = np.mean(results_train)
result_final_test = np.mean(results_test)
                     
print(result_final_train)
print(result_final_test)      
      
      
######################################################################################################
# PRODUCCION

X_train_T = X[0:20000]
y_train_T = y
X_summit_T = X[20000:]

from sklearn.neural_network import MLPClassifier
classifier_to_pred = MLPClassifier(random_state = 1234 , max_iter = 50 , learning_rate_init=0.05,
                           alpha = 0.1)

classifier_to_pred = classifier_to_pred.fit(X_train_T,y_train_T)

# Predict Train
y_pred_proba_train = pd.DataFrame(classifier_to_pred.predict_proba(X_train_T), columns = ["Proba_1","Proba_2","Proba_3","Proba_4"])
pred_train = classifier_to_pred.predict(X_train_T)

# Predict Test
y_pred_proba_PROD = pd.DataFrame(classifier_to_pred.predict_proba(X_summit_T), columns = ["Proba_1","Proba_2","Proba_3","Proba_4"])
pred_test = classifier_to_pred.predict(X_summit_T)

# To export
y_pred_proba_train_df = pd.DataFrame(y_pred_proba_train, columns = ["Proba_1","Proba_2","Proba_3","Proba_4"])
TRAIN_EXPORT = pd.concat([train_ori["COD_ENCUESTADO"],y_pred_proba_train_df] , axis = 1)
TRAIN_EXPORT = pd.concat([TRAIN_EXPORT,pd.DataFrame(pred_train,columns = ['Pred_Words'])] , axis = 1)
TRAIN_EXPORT.to_csv("data_words_NPL.csv", index = False)

y_pred_proba_test_df = pd.DataFrame(y_pred_proba_PROD, columns = ["Proba_1","Proba_2","Proba_3","Proba_4"])
TEST_EXPORT = pd.concat([test_ori["COD_ENCUESTADO"],y_pred_proba_test_df] , axis = 1)
TEST_EXPORT = pd.concat([TEST_EXPORT,pd.DataFrame(pred_test,columns = ['Pred_Words'])] , axis = 1)
TEST_EXPORT.to_csv("data_words_PROD_NPL.csv", index = False)










