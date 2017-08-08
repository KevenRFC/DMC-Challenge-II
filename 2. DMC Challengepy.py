
#Importar Librarias
import pandas as pd
import numpy as np

# Cargar Datos
train_ori = pd.read_excel("E:/MODELOS_DS/COMPETENCIAS NACIONALES/DMC CHALLENGE/train_universidad.xlsx")
test_ori = pd.read_excel("E:/MODELOS_DS/COMPETENCIAS NACIONALES/DMC CHALLENGE/test_universidad.xlsx")

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
train = pd.DataFrame(train_ori.values, columns = train_ori.columns)
test = pd.DataFrame(test_ori.values, columns = test_ori.columns)

train.drop("COMENTARIO",axis = 1, inplace = True)
test.drop("COMENTARIO",axis = 1, inplace = True)

## Tratamiento de valores nulos:
    # Training
train.isnull().sum() #Campos con Nulos

train.loc[-train["IND_GEA"].isnull(),"IND_GEA"] = 1
train.loc[ train["IND_GEA"].isnull(),"IND_GEA"] = 0
          
train.loc[-train["IND_DELEGADO"].isnull(),"IND_DELEGADO"] = 1
train.loc[ train["IND_DELEGADO"].isnull(),"IND_DELEGADO"] = 0

train.loc[-train["IND_DEPORT_CALIF"].isnull(),"IND_DEPORT_CALIF"] = 1
train.loc[ train["IND_DEPORT_CALIF"].isnull(),"IND_DEPORT_CALIF"] = 0

train['CANT_CURSOS'].fillna(0 , inplace = True) 
    
    # Testing
test.isnull().sum() #Campos con Nulos

test.loc[-test["IND_GEA"].isnull(),"IND_GEA"] = 1
test.loc[ test["IND_GEA"].isnull(),"IND_GEA"] = 0
          
test.loc[-test["IND_DELEGADO"].isnull(),"IND_DELEGADO"] = 1
test.loc[ test["IND_DELEGADO"].isnull(),"IND_DELEGADO"] = 0

test.loc[-test["IND_DEPORT_CALIF"].isnull(),"IND_DEPORT_CALIF"] = 1
test.loc[ test["IND_DEPORT_CALIF"].isnull(),"IND_DEPORT_CALIF"] = 0

test['CANT_CURSOS'].fillna(-1 , inplace = True) 

## Convertir a Numerico
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train["NIVEL_ACTUAL"] = le.fit_transform(train["NIVEL_ACTUAL"])

test["NIVEL_ACTUAL"] = le.transform(test["NIVEL_ACTUAL"])


## UNIFICAR VARIABLES RELACION:

train["RELAC_UNIV"] =  np.nan

train.loc[:,"RELAC_UNIV"] = train['IND_GEA'] + train['IND_DELEGADO'] + train['IND_DEPORT_CALIF'] 

train.loc[train["RELAC_UNIV"] >= 1,"RELAC_UNIV"] = 1
           
           
test["RELAC_UNIV"] =  np.nan

test.loc[:,"RELAC_UNIV"] = test['IND_GEA'] + test['IND_DELEGADO'] + test['IND_DEPORT_CALIF'] 

test.loc[test["RELAC_UNIV"] >= 1,"RELAC_UNIV"] = 1


### TRANFORMACIONES:
# Nombre Campus
train.loc[ [(boleano in [2,3]) for boleano in train['NIVEL_ACTUAL']] , 'NIVEL_ACTUAL'] = 4
test.loc[ [(boleano in [2,3]) for boleano in test['NIVEL_ACTUAL']] , 'NIVEL_ACTUAL'] = 4

# Clave de carrera
union_carreras = [18,103,38,8,22,102,222,40,1,223,111,213,214,220,210,217,219,224,218,216,19,225,
                  23,215,211,221,206,205,209,226,203]

train.loc[ [(boleano in union_carreras) for boleano in train['CLAVE_CARRERA']] , 'CLAVE_CARRERA'] = 300
test.loc[ [(boleano in union_carreras) for boleano in test['CLAVE_CARRERA']] , 'CLAVE_CARRERA'] = 300


# Ciclo
train.loc[ [(boleano in [10,11,12,13,14]) for boleano in train['CICLO']] , 'CICLO'] = 15
test.loc[ [(boleano in [10,11,12,13,14]) for boleano in test['CICLO']] , 'CICLO'] = 15

# Cantidad de concursos matriculados menos el ingles
train.loc[ [(boleano in [8,9,10]) for boleano in train['CANT_CURSOS']] , 'CANT_CURSOS'] = 11
test.loc[ [(boleano in [8,9,10]) for boleano in test['CANT_CURSOS']] , 'CANT_CURSOS'] = 11



# Separar Predictores y Target:
list(train.columns)
predictores =['NIVEL_ACTUAL',
              'CLAVE_CARRERA',
              'CICLO',
              #'IND_GEA',
              #'IND_DELEGADO',
              #'IND_DEPORT_CALIF',
              'CANT_CURSOS',
              'RELAC_UNIV'
              ]
target =  ['NPS']   
X_train = train[predictores]
y_train = np.array(train[target].astype("int")).ravel()

X_test = test[predictores]


################################## AGREGAR RESULTADOS PREDICCION NPL ########################## 
# Cargar Datos
train_ori_NPL = pd.read_csv("E:/MODELOS_DS/COMPETENCIAS NACIONALES/DMC CHALLENGE/bases_temp/data_words_NPL.csv")
test_ori_NPL = pd.read_csv("E:/MODELOS_DS/COMPETENCIAS NACIONALES/DMC CHALLENGE/bases_temp/data_words_PROD_NPL.csv")

X_train = pd.concat([X_train,train_ori_NPL[["Proba_1","Proba_2","Proba_3","Proba_4","Pred_Words"]]] , axis = 1) 
X_test = pd.concat([X_test,test_ori_NPL[["Proba_1","Proba_2","Proba_3","Proba_4","Pred_Words",]]] , axis = 1)

# Variales
temp = list(X_train.columns); temp

columns_to_keep = [#'NIVEL_ACTUAL',
 #'CLAVE_CARRERA',
 #'CICLO',
 #'CANT_CURSOS',
 #'RELAC_UNIV',
 'Proba_1',
 'Proba_2',
 'Proba_3',
 'Proba_4',
 'Pred_Words']

X_train = X_train[columns_to_keep]
X_test = X_test[columns_to_keep]

###################################### MODELAMIENTO ########################################## 

# Entrenar el modelo: RANDOM FOREST  
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 9876, min_samples_leaf = 100, n_estimators = 60)

classifier = classifier.fit(X_train, y_train)

# Predecir sobre el Training Dataset
y_pred_proba = pd.DataFrame(classifier.predict_proba(X_train), columns = ["NPS_1","NPS_2","NPS_3","NPS_4"])
y_pred = classifier.predict(X_train)


# Evaluacion del Modelo: 
    ## Matriz de Confusión
from sklearn import metrics as mt
cm = mt.confusion_matrix(y_train, y_pred)
print(cm)
accuracy = mt.accuracy_score(y_train,y_pred)
print ("Accuracy : ", round(accuracy,3)) # Accuracy : 

    # log_loss
from sklearn.metrics import log_loss
log_loss_pred = log_loss(y_train,y_pred_proba)
print ("log loss : ", round(log_loss_pred,3)) # log loss : 


'''
# Get Variables Important
importancia_variables = pd.concat([pd.DataFrame(classifier.feature_importances_  , columns = [0]),
                                   pd.DataFrame(X_train.columns , columns = [1])] , 
                                   axis = 1)
'''

# Cros Validation
from sklearn import cross_validation
from sklearn.metrics import log_loss
cv = cross_validation.KFold(len(X_train), n_folds=10, random_state = 0)

results_train = []
results_test = []

for traincv, testcv in cv:
        probas_train = classifier.fit(X_train.values[traincv], y_train[traincv]).predict_proba(X_train.values[traincv])
        probas_test = classifier.fit(X_train.values[traincv], y_train[traincv]).predict_proba(X_train.values[testcv])
        log_loss_pred_train = log_loss(y_train[traincv],probas_train)
        log_loss_pred_test = log_loss(y_train[testcv],probas_test)
        results_train.append(log_loss_pred_train)
        results_test.append(log_loss_pred_test)
        print("Ok: ", str(log_loss_pred_train), str(log_loss_pred_test))

result_final_train = np.mean(results_train)
result_final_test = np.mean(results_test)

print(result_final_train) # 0.750828327947
print(result_final_test) #  0.775446236322
                    
######################################################################################################
# BASE PRODUCCION:
    
model_final = RandomForestClassifier(random_state = 9876, min_samples_leaf = 100, n_estimators = 60)
model_final = model_final.fit(X_train.values, np.array(y_train).ravel())

temp_pred_prod = model_final.predict_proba(X_test.values)
y_pred_proba_PROD = pd.DataFrame(temp_pred_prod, columns = ["NPS1","NPS2","NPS3","NPS4"])
data_summit = pd.concat([test["COD_ENCUESTADO"],y_pred_proba_PROD] , axis = 1)
data_summit.to_csv("summit_new.csv", 
                   index = False)
     
     
     