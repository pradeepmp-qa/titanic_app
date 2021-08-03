##### Importing Relevant Libraries ######
#########################################

import pandas as pd
import numpy as np
import os
import seaborn as sns
import pickle ### helps storing data in pickle files

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, plot_roc_curve, confusion_matrix,f1_score ## model evaluation metrics
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler ## (x-mean(variable))/standard_deviation(variable)
from sklearn.model_selection import StratifiedShuffleSplit ### datasplitting
 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV          ### hyperparameter finding

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor          ### helps fitting a decision tree
from sklearn.ensemble import RandomForestClassifier       ### models from sklearn
from sklearn.svm import SVC

####### Loading Data ######
###########################

cwd = os.getcwd() ##gets current working directory

data_path = str(cwd) + '/train.csv'

data_train = pd.read_csv(data_path)  ## Loads the data
data_test = pd.read_csv('test.csv')

print(data_train)

print(data_test)

print(data_train.info())

#### Insights ####
# PassengerId, Name could be removed 
# Cabin has 687 row data missing and hence we cannot infer much from this feature and better to be removed
# Age has 177 row data missing we will have to further analyse and impute the data appropriately 
# Sex, Ticket, Embarked to be converted to numeric data

# lets combine the data for data prep

data_test['Survived'] = np.nan

data_train["data"] = 'train'
data_test["data"] = 'test'

data_all = pd.concat([data_train,data_test],axis = 0)

# Have now concatenated 891 data rows from train and 418 data rows from test and have 1309 in total
print(data_all)

print(data_all.shape, data_train.shape, data_test.shape)

# PassengerId, Name could be removed 
# Cabin has 687 row data missing and hence we cannot infer much from this feature and better to be removed
# Age has 177 row data missing we will have to further analyse and impute the data appropriately 
# Sex, Ticket, Embarked to be converted to numeric data
print(data_all.dtypes)

data_all.drop(['PassengerId','Name','Cabin'], axis = 'columns', inplace = True)

print(data_all.info())

#### Visualizing numeric columns ######

# numeric_cols = data_all.select_dtypes(include = np.number) ### selects numeric columns

# column_names = list(numeric_cols.columns)

# col_index = 0

# plot_rows = 2
# plot_cols = 3

# fig, ax = plt.subplots(nrows = plot_rows,ncols=plot_cols,figsize = (20,10))

# for row_count in range(plot_rows):
#     for col_count in range(plot_cols):
#         ax[row_count][col_count].scatter(y = numeric_cols[column_names[col_index]],x=numeric_cols.index)
#         ax[row_count][col_count].set_ylabel(column_names[col_index])
#         col_index = col_index + 1


genuine_numeric_cols = ['Survived','Pclass','Age','SibSp','Parch','Fare']

numeric_cols = data_all.loc[data_all['data']=='train',genuine_numeric_cols] ### selects numeric columns

column_names = list(numeric_cols.columns)

col_index = 0

plot_rows = 3
plot_cols = 2

fig, ax = plt.subplots(nrows = plot_rows,ncols=plot_cols,figsize = (20,10))

for row_count in range(plot_rows):
    for col_count in range(plot_cols):
        ax[row_count][col_count].scatter(y = numeric_cols[column_names[col_index]],x=numeric_cols.index)
        ax[row_count][col_count].set_ylabel(column_names[col_index])
        col_index = col_index + 1

# Capping outliers in Fare which are in ~500 range to 99.7 percentile value 
data_all.loc[data_all['data']=='train','Fare'].quantile(0.997)
data_all.loc[data_all['Fare'] >= 400,'Fare'] = 345
data_all.loc[data_all['Fare'].isnull(),'Fare'] = 345

data_all.loc[data_all['Embarked'].isnull(),'Embarked'] = 'S'

# Imputing Age for missing values
data_all.loc[data_all['data'] == 'train','Age'].mean()
data_all.loc[data_all['Age'].isnull(),'Age'] = 29.6

# Setting dummy values for Sex Columns
Sex_data = pd.get_dummies(data_all['Sex'],prefix = 'Sex')

data_all.drop(['Sex'],axis = 1, inplace = True)

data_all = pd.concat([data_all,Sex_data],axis = 1)

# Dropping Ticket column as there may be no much relevance
data_all.drop(['Ticket'],axis = 1, inplace = True)

# Setting dummy values for Embarked Columns
Embarked_data = pd.get_dummies(data_all['Embarked'],prefix = 'Embarked', drop_first = True)

data_all.drop(['Embarked'],axis = 1, inplace = True)

data_all = pd.concat([data_all,Embarked_data],axis = 1)

print(data_all)

####### Preprocess-5 : Scaling the columns ######

column_names = ['Pclass','Age','SibSp','Parch','Fare']
scaler = StandardScaler()  ### instance of this object

scaler.fit(data_all[column_names]) ### it will compute mean and standard deviation of every column

data_all[column_names] = scaler.transform(data_all[column_names]) #### apply the formula (x-mean)/s.d

print(data_all)

data_train=data_all.loc[data_all['data']=='train']
data_train.drop(['data'],axis = 1, inplace = True)
data_train

print(data_all)

data_test=data_all.loc[data_all['data']=='test']

data_test.drop(['Survived','data'],axis=1,inplace=True)

print(data_train)

print(data_test)

target_train = data_train['Survived']
target_train

# Converting flot to int for all the target values to be inline with original dataset
target_train = target_train.astype(int) 
print(target_train)

data_train.drop('Survived',axis = 1, inplace = True)

from sklearn.model_selection import train_test_split ### Help me to split the data into train and validation

X_train, X_test, y_train, y_test = train_test_split(data_train,target_train,test_size=0.2,random_state=100)

print(X_train)

model_params = {
    
    'SVC' : {
        'model' : SVC(gamma = 'auto'),
        'params' : {
            'C' : [10,20,30],
            'kernel': ['rbf','linear']
            }
    },
    
    'random_forest' : {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators':[1,5,10]
        }
    },
    
    'logistic_regression': {
        'model': LogisticRegression(solver = 'liblinear'),
        'params' : {
            'max_iter':[50,100]
            
        }
        
    },
        'decision_tree' : {
            'model': DecisionTreeClassifier(),
            'params' : {
                'criterion':["gini", "entropy"],
                'max_depth':[20,40,60]
            }

        }

}


scores = []
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'],mp['params'], cv = 5)
    clf.fit(X_train,y_train)
    scores.append({
        'model' : model_name,
        'best_score': clf.best_score_,
        'best_params' : clf.best_params_
        
    }   
    
    )


df = pd.DataFrame(scores,columns = ['model','best_score','best_params'])
print(df)

clf = SVC(C = 20,kernel = 'rbf')

clf.fit(X_train,y_train)

### Evaluating on the train and the test set ####
predicted_train = clf.predict(X_train)

plot_roc_curve(clf,X =X_train, y= y_train)

print ('The score for the SVC model ', roc_auc_score(y_train,predicted_train))

print(confusion_matrix(y_true = y_train, y_pred = predicted_train))

print ('The F1-SCORE on the train set prediction ',f1_score(y_true=y_train,y_pred = predicted_train,sample_weight = y_train))

plot_roc_curve(clf,X =X_test, y= y_test)

predicted_test = clf.predict(X_test)

print (confusion_matrix(y_true = y_test, y_pred = predicted_test))

print ('The F1-SCORE on the test set prediction ',f1_score(y_true=y_test,y_pred = predicted_test,sample_weight = y_test))

# Fitting model with entire 891 data set

new_clf = SVC(C = 20,kernel = 'rbf')
new_clf.fit(data_train,target_train)

### Evaluating on the train complete data - 891 data set
predicted_train = new_clf.predict(data_train)

plot_roc_curve(new_clf,X =data_train, y= target_train)

print ('The score for the SVC model ', roc_auc_score(target_train,predicted_train))

print(confusion_matrix(y_true = target_train, y_pred = predicted_train))

print ('The F1-SCORE on the train set prediction ',f1_score(y_true=target_train,y_pred = predicted_train,sample_weight = target_train))

print(data_test)

with open('SVCmodel_new_clf.pickle', 'wb') as f:
    pickle.dump(new_clf,f)











































