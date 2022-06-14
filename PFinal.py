#%%
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
import pandas as pd
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import math
import seaborn as sn
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Para escalar lo datos
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 500)

seed = 3986
#%%

print("Leyendo los datos ...")
df = pd.read_excel("./datos/data.xls",header=0, skiprows=1, index_col=0)

print(df.head())
# Obtenemos la X y la Y
y = df.iloc[:, 23]
X = df.iloc[:, :23]


print("Número de datos: ", X.shape[0])
print("Número de atributos: ", X.shape[1])

# Transformamos la variable educación de manera que se encuentre dividida
# en 4 categorías como se indica en UCI y no en 6 como se encuntra realmente.
# Esto se hace uniendo la clase minoritaria en una clase a la que llamaremos
# "others"
X['EDUCATION'] = X['EDUCATION'].where(X['EDUCATION'].isin([1, 2, 3]), 4)
# Lo mismo con la variable de estado civil
X['MARRIAGE'] = X['MARRIAGE'].where(X['MARRIAGE'].isin([1, 2]), 3)

#%%

print("Particionando los datos...")
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=seed)
print("Número de datos de entrenamiento: ", len(y_train))
print("Número de datos de test: ", len(y_test))
print("")
# Descripción de las variables continuas
print("Descripción de las variables continuas del conjunto de train: ")
print(X_train.describe().T)
# %%

# Función que muestra un histograma de la columna de un dataframe
def hist_plot(df, column, bins = None, rng = None, ylabel = 'Count'):
    plt.hist(X[column], bins=bins, range=rng)
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel(ylabel)
    plt.show()
    
def bar_plot(serie, xticks_labels=None, title=None):
    sn.countplot(x = serie)
    if title == None:
        plt.title("Bar plot of var " + serie.name)
    else:
        plt.title(title)
    if(xticks_labels != None):
        # ticks = serie.unique()
        # ticks = np.sort(ticks)
        ticks = np.arange(0, len(xticks_labels))
        plt.xticks(ticks=ticks, labels=xticks_labels)
    plt.show()

def hist_plot(serie, bins = None, rng = None, ylabel = 'Count'):
    plt.hist(serie, bins=bins, range=rng)
    plt.title(serie.name)
    plt.xlabel(serie.name)
    plt.ylabel(ylabel)
    plt.show()

# sn.countplot(x=y)
# plt.title("Número de \"si\" frente a \"no\"")
# plt.xticks(ticks=[0, 1], labels=["Pago", "Impago"])
# plt.show()

bar_plot(y_train,
         xticks_labels=["Pago", "Impago"],
         title="Número de pagos frente a impagos")

bar_plot(X_train['SEX'],
         xticks_labels=['Male', 'Female'])

bar_plot(X_train['EDUCATION'],
         xticks_labels=['graduate school', 'university', 'high school', 'others'])

bar_plot(X_train['MARRIAGE'],
         xticks_labels=['Married', 'Single', 'Others'])
        
hist_plot(X_train['AGE'])

# %%

def to_onehot(df, column):
    '''
    Transforma una columna del dataframe a one-hot encoding
    Parameters
    ----------
    df : DataFrame
        Dataframe.
    column : String
        Nombre de la columna a transformar.

    Returns
    -------
    sub_df : DataFrame
        Dataframe pasado con la codificación de la columna en formato
        one-hot encoding.

    '''
    sub_df = df[column]
    dum_df = pd.get_dummies(sub_df, prefix = column)
    sub_df = df.join(dum_df)
    sub_df = sub_df.drop(columns=[column])
    return sub_df

X_train = to_onehot(X_train, 'SEX')
X_train = to_onehot(X_train, 'EDUCATION')
X_train = to_onehot(X_train, 'MARRIAGE')

X_test = to_onehot(X_test, 'SEX')
X_test = to_onehot(X_test, 'EDUCATION')
X_test = to_onehot(X_test, 'MARRIAGE')
# X_train = to_onehot(X_train, 'PAY_0')
# X_train = to_onehot(X_train, 'PAY_2')
# X_train = to_onehot(X_train, 'PAY_3')
# X_train = to_onehot(X_train, 'PAY_4')
# X_train = to_onehot(X_train, 'PAY_5')
# X_train = to_onehot(X_train, 'PAY_6')

print("Nuevo número de variables: ", X_train.shape[1])



# Escalamos las variables no binarias
# Para ello primero separamos los datos en train y test
# y los escalaremos todos en función de los de train
def standard_scale_vars(train_data, test_data, variables_to_scale):
    # Iniciamos el scalador
    scaler = StandardScaler()
    # Lo ajustamos a los datos de train
    scaler.fit(train_data[variables_to_scale])
    # Transformamos train y test con el escalador
    train_data[variables_to_scale] = scaler.transform(train_data[variables_to_scale])
    test_data[variables_to_scale] = scaler.transform(test_data[variables_to_scale])

    
    return train_data, test_data
    
# Variables a escalar. Las binarias no será necesario escalarlas
variables_to_scale = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3',
                      'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                      'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                      'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4',
                      'PAY_AMT5', 'PAY_AMT6']

X_train, X_test = standard_scale_vars(X_train, X_test, variables_to_scale)

#%%

params = [
            {
                "penalty" : ['l1', 'l2'], # Tipo de regularización
                "alpha" : [0.0001, 0.001, 0.01, 0.1], # Factor de regularización
                "max_iter" : [10000],
                "learning_rate": ['adaptive'], # Política de actualización del lr
                "eta0" : [0.0001, 0.001, 0.01, 0.1], # Learning rate
                # "l1_ratio" : [0.0, 0.30, 0.5, 0.7, 1.0] # Ratio entre L1 y L2
            }
]


# Cross validator para pasarlo a GridSearchCV
cv = KFold(n_splits=5, shuffle=False)

best_estimators_rl = {}

metrica = 'recall'
print("------------ Regresión Logística ------------")
# Creamos el estimador
estimador = SGDClassifier(loss="log", 
                          class_weight = "balanced",
                          early_stopping=False, 
                          n_jobs=-1,
                          random_state=seed)        
# Creamos el GridSearchCV
clf = GridSearchCV(estimator = estimador,
                   param_grid = params,
                   cv = cv, 
                   n_jobs = -1, scoring = metrica)
# Lo entrenamos con los datos de train
clf.fit(X_train, y_train)
print("Mejor resultado: ", clf.best_score_)
print("Mejores parámetros: ", clf.best_params_)
print("")
best_estimators_rl[metrica] = clf.best_estimator_

y_pred = clf.predict(X_test)
print("Test score: ", metrics.f1_score(y_test, y_pred))
print("Test score: ", metrics.recall_score(y_test, y_pred))

# Dibujamos la matriz de confusión
def matriz_confusion(y_true, y_pred, title = None):
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    sn.heatmap(confusion_matrix, annot = True, fmt='d')
    plt.title(title)
    plt.xlabel('predicción')
    plt.ylabel('valor real')
    plt.show()

matriz_confusion(y_test, y_pred)



#%%

# Multilayer Perceptron
print("------------ Multilayer Perceptron ------------")


clf = MLPClassifier(hidden_layer_sizes=(100,),
                    activation='relu',
                    solver='adam',
                    learning_rate_init=0.0001,
                    max_iter=1000,
                    batch_size=64
                    )
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print("Train Accuracy: ", metrics.accuracy_score(y_train, y_pred))
y_pred = clf.predict(X_test)
print("Test Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("Test f1: ", metrics.recall_score(y_test, y_pred))
print("Test f1: ", metrics.recall_score(y_test, y_pred))

#%%

    
matriz_confusion(y_test, y_pred)





# print(X_train.describe().T)