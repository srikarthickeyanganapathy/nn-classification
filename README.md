# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![309490523-403623cf-4a09-4bb5-9636-b47447109ecd](https://github.com/SanthoshUthiraKumar/nn-classification/assets/119477975/dc86aa7c-8799-4e5c-a96d-98431db787ad)

## DESIGN STEPS

## STEP 1:
Import the necessary packages & modules

## STEP 2:
Load and read the dataset

## STEP 3:
Perform pre processing and clean the dataset

## STEP 4:
Encode categorical value into numerical values using ordinal/label/one hot encoding

## STEP 5:
Visualize the data using different plots in seaborn

## STEP 6:
Normalize the values and split the values for x and y

## STEP 7:
Build the deep learning model with appropriate layers and depth

## STEP 8:
Analyze the model using different metrics

## STEP 9:
Plot a graph for Training Loss, Validation Loss Vs Iteration & for Accuracy, Validation Accuracy vs Iteration

## STEP 10:
Save the model using pickle

## STEP 11:
Using the DL model predict for some random inputs


## PROGRAM

### Name: SRI KARTHICKEYAN GANAPATHY
### Register Number: 212222240102

```py
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt

df = pd.read_csv('/customers.csv')
df.head()
df.columns
df.dtypes
df.shape
df.isnull().sum()
df_cleaned=df.dropna(axis=0)
df_cleaned.isnull().sum()
df_cleaned.shape
df_cleaned.dtypes
df_cleaned['Gender'].unique()
df_cleaned['Ever_Married'].unique()
df_cleaned['Graduated'].unique()
df_cleaned['Family_Size'].unique()
df_cleaned['Var_1'].unique()
df_cleaned['Spending_Score'].unique()
df_cleaned['Profession'].unique()
df_cleaned['Segmentation'].unique()

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

categories_list=[['Male','Female'],
                 ['No','Yes'],
                 ['No','Yes'],
                 ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
                 'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
                 ['Low','Average','High']
                 ]
enc = OrdinalEncoder(categories = categories_list)

cust_1=df_cleaned.copy()
cust_1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']]=enc.fit_transform(cust_1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']])

cust_1.dtypes


le = LabelEncoder()
     

cust_1['Segmentation'] = le.fit_transform(cust_1['Segmentation'])
     

cust_1.dtypes
     

cust_1 = cust_1.drop('ID',axis=1)
cust_1 = cust_1.drop('Var_1',axis=1)
     

cust_1.dtypes
     

# Calculate the correlation matrix
corr = cust_1.corr()

# Plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)

sns.pairplot(cust_1)

cust_1.describe()
cust_1['Segmentation'].unique()

X=cust_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values

y1 = cust_1[['Segmentation']].values
one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
y1.shape
y = one_hot_enc.transform(y1).toarray() 
y.shape    
y1[0]
y[0]
X.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)
X_train[0]
X_train.shape
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

# To scale the Age column
X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)


# Creating the model
ai_brain = Sequential([
  Dense(4,input_shape=(8,)),
  Dense(8,activation='relu'),
  Dense(8,activation='relu'),
  Dense(4,activation='softmax'),
])

ai_brain.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2)


ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs=2000,batch_size=256,
             validation_data=(X_test_scaled,y_test),
             )
metrics = pd.DataFrame(ai_brain.history.history)

metrics.head()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
     
# Saving the Model
ai_brain.save('customer_classification_model.h5')

# Saving the data
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,cust_1,df_cleaned,scaler_age,enc,one_hot_enc,le], fh)

ai_brain = load_model('customer_classification_model.h5')

# Loading the data
with open('customer_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,customers_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)
     
#Prediction for a single input
x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))


```

## Dataset Information

![309490559-8d0331ed-9f34-4cba-a1e8-6239468e2ba2](https://github.com/SanthoshUthiraKumar/nn-classification/assets/119477975/7fe8d5fd-b788-49c5-99ff-72941519ce3d)


## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

![1 1](https://github.com/SanthoshUthiraKumar/nn-classification/assets/119477975/6f7b0322-66cd-4b36-a3ec-1440a236523a)

### Classification Report

![1 3](https://github.com/SanthoshUthiraKumar/nn-classification/assets/119477975/9f1d031a-8fba-41b5-9cc0-b612130c3720)

### Confusion Matrix

![1 2](https://github.com/SanthoshUthiraKumar/nn-classification/assets/119477975/74582b10-9ea2-4f74-9a91-e08b813f9033)


### New Sample Data Prediction

![1 4](https://github.com/SanthoshUthiraKumar/nn-classification/assets/119477975/fbd3b9ae-49c4-4861-984c-3e207808a0aa)

### RESULT
A neural network classification model is developed for the given dataset.



