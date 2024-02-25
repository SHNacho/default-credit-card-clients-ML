#%%
from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
import pandas as pd
from sklearn.linear_model import SGDClassifier, Perceptron, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.model_selection import learning_curve, ShuffleSplit, cross_validate
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import matplotlib.pyplot as plt
import math
import seaborn as sn
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Para escalar lo datos
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 500)

seed = 3986

####################################################################
# DEFINICIÓN DE MÉTODOS

def hist_plot(serie, bins = None, rng = None, ylabel = 'Count'):
    plt.hist(serie, bins=bins, range=rng)
    plt.title(serie.name)
    plt.xlabel(serie.name)
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

# Escala las variables de train y test en función de los datos de train
def standard_scale_vars(train_data, test_data, variables_to_scale):
    # Iniciamos el scalador
    scaler = StandardScaler()
    # Lo ajustamos a los datos de train
    scaler.fit(train_data[variables_to_scale])
    # Transformamos train y test con el escalador
    train_data[variables_to_scale] = scaler.transform(train_data[variables_to_scale])
    test_data[variables_to_scale] = scaler.transform(test_data[variables_to_scale])

    
    return train_data, test_data

def matriz_confusion(y_true, y_pred, title = None):
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    sn.heatmap(confusion_matrix, annot = True, fmt='d')
    plt.title(title)
    plt.xlabel('predicción')
    plt.ylabel('valor real')
    plt.show()
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
# Descripción de las variables
print("Descripción de las variables del conjunto de train: ")
print(X_train.describe().T)
# print(X_train.describe().T[['count', 'mean', 'std']].to_latex(float_format="%.2f", bold_rows=True))
# %%
##############################
# VISUALIZACIÓN DE LOS DATOS #
##############################

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

hist_plot(X_train['LIMIT_BAL'], bins=50)

for column in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
    bar_plot(X_train[column])

for column in ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']:
    hist_plot(X_train[column], bins=50, rng=(0, 250000))
    
for column in ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']:
    hist_plot(X_train[column], bins=50, rng=(0, 75000))

# %%
###############################
# TRANSFORMACIÓN DE LOS DATOS #
###############################

# Pasamos las variables necesarias a codificación one-hot en trai y test
X_train = to_onehot(X_train, 'SEX')
X_train = to_onehot(X_train, 'EDUCATION')
X_train = to_onehot(X_train, 'MARRIAGE')

X_test = to_onehot(X_test, 'SEX')
X_test = to_onehot(X_test, 'EDUCATION')
X_test = to_onehot(X_test, 'MARRIAGE')

print("Nuevo número de variables: ", X_train.shape[1])
    
# Variables a escalar. Las binarias no será necesario escalarlas
variables_to_scale = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3',
                      'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                      'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                      'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4',
                      'PAY_AMT5', 'PAY_AMT6']

X_train, X_test = standard_scale_vars(X_train, X_test, variables_to_scale)

# print(X_train.describe().T[['count', 'mean', 'std']].to_latex(float_format="%.2f", bold_rows=True))

# #%%
# ################################
# # SELECCIÓN DE CARACTERÍSTICAS #
# ################################

# # Usaremos regresión lasso para la selección de características
# params = [
#             {
#                 "alpha" : np.logspace(10e-4, 1)
#             }
# ]

# lasso = Lasso()

# clf = GridSearchCV(estimator=lasso,
#                    param_grid=params,
#                    cv=5,
#                    n_jobs=-1)

# clf.fit(X_train, y_train)
# print("Resultados grid search: ")
# print("Mejor resultado: ", clf.best_score_)
# print("Mejores parámetros: ", clf.best_params_)
# print("")

# #%%
# # Como el mejor alpha me da 1.0
# lasso = Lasso(alpha=1.0)
# lasso.fit(X_train, y_train)
# model = SelectFromModel(lasso, prefit=True)
# X_new = model.transform(X_train)
# print(X_new.shape)

#%%
# Selección de la métrica
# metrica = metrics.recall_score
metrica = metrics.f1_score
# metrica = metrics.accuracy_score

#%%

############################################################################
############################ LOGISTIC REGRESION ############################
############################################################################

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

print("------------ Regresión Logística ------------")
# Creamos el estimador
estimador = SGDClassifier(loss="log", 
                          class_weight = 'balanced',
                          early_stopping=False, 
                          n_jobs=-1,
                          random_state=seed)        
# Creamos el GridSearchCV
clf = GridSearchCV(estimator = estimador,
                   param_grid = params,
                   cv = cv, 
                   n_jobs = -1, scoring = 'f1')
# Lo entrenamos con los datos de train
clf.fit(X_train, y_train)
print("Resultados grid search: ")
print("Mejor resultado: ", clf.best_score_)
print("Mejores parámetros: ", clf.best_params_)
print("")

y_pred_rl_train = clf.predict(X_train)
y_pred_rl_test = clf.predict(X_test)
print("Ein = ", metrica(y_train, y_pred_rl_train))
print("Eout = ", metrica(y_test, y_pred_rl_test))

# Dibujamos la matriz de confusión
matriz_confusion(y_test, y_pred_rl_test)



#%%

############################################################################
########################## MULTILAYER PERCEPTRON ###########################
############################################################################

print("------------ Multilayer Perceptron ------------")

params = [
            {
                "learning_rate_init" : [0.0001, 0.001, 0.01, 0.1],
            }
]


mlp = MLPClassifier(hidden_layer_sizes=(100,),
                    activation='relu',
                    solver='adam',
                    learning_rate_init=0.0001,
                    max_iter=1000,
                    batch_size=64
                    )

clf = GridSearchCV(estimator=mlp, 
                   param_grid=params,
                   scoring='f1',
                   cv=cv,
                   n_jobs=-1)

clf.fit(X_train, y_train)

print("Resultados tras grid search sobre el parámetro learning_rate: ")
print("Mejor resultado: ", clf.best_score_)
print("Mejores parámetros: ", clf.best_params_)


clf.fit(X_train, y_train)
y_pred_mlp_train = clf.predict(X_train)
y_pred_mlp_test = clf.predict(X_test)

print("Ein = ", metrica(y_train, y_pred_mlp_train))
print("Eout = ", metrica(y_test, y_pred_mlp_test))

    
matriz_confusion(y_test, y_pred_mlp_test)

#%%

############################################################################
############################## RANDOM FOREST ###############################
############################################################################
print("------------ Random Forest ------------")

# Ajuste del parámetro max_features
# https://stats.stackexchange.com/questions/111968/random-forest-how-to-handle-overfitting
params = [
            {
                # "max_features" : [0.1, 0.3, 0.5, 0.7, 0.9],
                "max_features" : [0.3],
            }
]

random_forest = RandomForestClassifier(n_estimators=200,
                                       max_depth=30,
                                       min_samples_leaf=1000,
                                       class_weight='balanced')

cv = KFold(n_splits=5, shuffle=False)

clf = GridSearchCV(estimator = random_forest,
                   param_grid = params,
                   cv = cv, 
                   n_jobs = -1, scoring = 'f1')

clf.fit(X_train, y_train)

print("Resultados tras grid search sobre el parámetro max_features: ")
print("Mejor resultado: ", clf.best_score_)
print("Mejores parámetros: ", clf.best_params_)

y_pred_rf_train = clf.predict(X_train)
y_pred_rf_test = clf.predict(X_test)



print("Ein = ", metrica(y_train, y_pred_rf_train))
print("Eout = ", metrica(y_test, y_pred_rf_test))

matriz_confusion(y_test, y_pred_rf_test)





# print(X_train.describe().T)
