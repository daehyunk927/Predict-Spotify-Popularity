
# Import needed libraries
from collections import Counter
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#Read in Data
df = pd.read_csv('song_data.csv')

#Perform EDA
f,ax = plt.subplots(figsize=(12, 12))
mask = np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(df.corr(), annot=True, linewidths=0.4,linecolor="white", fmt= '.1f',ax=ax,cmap="Blues",mask=mask)
plt.show() 

f, axes = plt.subplots(3, 5, figsize=(15, 15))
sns.distplot( df["song_duration_ms"] , color="blue", ax=axes[0, 0])
sns.distplot( df["instrumentalness"] , color="blue", ax=axes[0, 1])
sns.distplot( df["acousticness"] , color="blue", ax=axes[0, 2])
sns.distplot( df["danceability"] , color="blue", ax=axes[0, 3])
sns.distplot( df["energy"] , color="blue", ax=axes[0, 4])
sns.distplot( df["song_popularity"] , color="blue", ax=axes[1, 0])
sns.distplot( df["key"] , color="blue", ax=axes[1, 1])
sns.distplot( df["liveness"] , color="blue", ax=axes[1, 2])
sns.distplot( df["loudness"] , color="blue", ax=axes[1, 3])
sns.distplot( df["audio_mode"] , color="blue", ax=axes[1, 4])
sns.distplot( df["tempo"] , color="blue", ax=axes[2, 0])
sns.distplot( df["speechiness"] , color="blue", ax=axes[2, 1])
sns.distplot( df["time_signature"] , color="blue", ax=axes[2, 2])
sns.distplot( df["audio_valence"] , color="blue", ax=axes[2, 3])
f.delaxes(axes[2][4])


plt.show()

#Perform necessary transformation to predictor variables
bins = [-40, -30, -20, -10, 0]
names = [1, 2, 3, 4]

df['loudness'] = pd.cut(df['loudness'], bins, labels=names)

#Select outcome variable; transform outcome variable
#We want our outcome variable to be categorical, so song popularity will be broken into 5 tiers
df = df.dropna()

df['song_popularity'] = (df['song_popularity'].astype(int))

bins = [0, 20, 40, 60, 80, 100]
names = [1, 2, 3, 4, 5]

df['popularity'] = pd.cut(df['song_popularity'], bins, labels=names)

df = df.dropna()

X = df.drop('popularity', 1)
X = X.drop('song_name', 1)
X = X.drop('song_popularity', 1)
Y = df.popularity
Y = Y.astype(int)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Convert train split to numpy arrays
X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_test = np.asarray(X_test)
Y_test = np.asarray(Y_test)

#Select Model
import itertools    

def make_param_grids(steps, param_grids):

    final_params=[]
    # (pca OR svd) AND (svm OR rf) will become ->
    # (pca, svm) , (pca, rf) , (svd, svm) , (svd, rf)
    for estimator_names in itertools.product(*steps.values()):
        current_grid = {}

        # Step_name and estimator_name should correspond
        # i.e preprocessor must be from pca and select.
        for step_name, estimator_name in zip(steps.keys(), estimator_names):
            for param, value in param_grids.get(estimator_name).items():
                if param == 'object':
                    # Set actual estimator in pipeline
                    current_grid[step_name]=[value]
                else:
                    # Set parameters corresponding to above estimator
                    current_grid[step_name+'__'+param]=value
        #Append this dictionary to final params            
        final_params.append(current_grid)

    return final_params

pipeline_steps = {'classifier':['svc', 'log', 'knn', 'percep', 'mlp','rf']}

# fill parameters to be searched in this dict
all_param_grids = {'svc':{'object':LinearSVC(), 
                          'penalty':['l2']
                         }, 
                   
                   'knn':{'object':KNeighborsClassifier()
                         }, 
                   
                   'percep':{'object':Perceptron(), 
                          'penalty':['l1','l2']
                         }, 

                   'log':{'object':LogisticRegression(),
                         'penalty':['l1','l2'],
                         'solver':['saga'],
                         'multi_class':['ovr', 'multinomial'] 
                         },
                   'mlp':{'object':MLPClassifier(),
                          'hidden_layer_sizes': [(500, ),(200,200),(100, )],
                          'activation': ['relu','logistic','tanh'],
                          'max_iter': [200, 300]
                         },
                   'rf': {'object': RandomForestClassifier(),
                          'n_estimators': [100,200],
                         }
                   
                  }  

# Try more models (non-linear ones): basic neural nets, SVC

# Call the method on the above declared variables
param_grids_list = make_param_grids(pipeline_steps, all_param_grids)

# The PCA() and SVC() used here are just to initialize the pipeline,
# actual estimators will be used from our `param_grids_list`
pipe = Pipeline(steps=[('classifier', LinearSVC())]) 

grd = GridSearchCV(pipe, param_grid = param_grids_list)
grd.fit(X_train,Y_train)
grd.best_estimator_

# Test Strength of Chosen Random Forest
# Random Forest

rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

rf.fit(X_train, Y_train)

y_pred = rf.predict(X_test)
y_true= Y_test

print("Accuracy of model is {0:.4f}".format(accuracy_score(y_true, y_pred)))
print("Precision of model is {0:.4f}".format(precision_score(y_true, y_pred, average='macro')))
print("Recall of model is {0:.4f}".format(recall_score(y_true, y_pred, average='macro')))
print("F1 score of model is {0: .4f}".format(f1_score(y_true, y_pred, average='macro')))

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(rf, open(filename, 'wb'))