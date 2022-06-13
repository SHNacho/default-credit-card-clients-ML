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

bar_plot(y,
         xticks_labels=["Pago", "Impago"],
         title="Número de pagos frente a impagos")

bar_plot(X['SEX'],
         xticks_labels=['Male', 'Female'])

bar_plot(X['EDUCATION'])
        #  xticks_labels=['graduate school', 'university', 'high school', 'others'])

bar_plot(X['MARRIAGE'])
        #  xticks_labels=['Married', 'Single', 'Others'])
        
hist_plot(X['AGE'])

# sn.countplot(x = X['SEX'])
# plt.title("Género")
# plt.xticks(ticks=[1, 2], labels=['Male', 'Female'])
# plt.show()
# %%
