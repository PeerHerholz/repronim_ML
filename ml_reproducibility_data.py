import urllib.request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


url = 'https://www.dropbox.com/s/v48f8pjfw4u2bxi/MAIN_BASC064_subsamp_features.npz?dl=1'
urllib.request.urlretrieve(url, 'MAIN2019_BASC064_subsamp_features.npz')

url = 'https://www.dropbox.com/s/ofsqdcukyde4lke/participants.csv?dl=1'
urllib.request.urlretrieve(url, 'participants.csv')

data = np.load('MAIN2019_BASC064_subsamp_features.npz')['a']
data.shape

labels = pd.read_csv('participants.csv')['AgeGroup']

cv = StratifiedKFold()

pipe = make_pipeline(
    StandardScaler(),
    RandomForestClassifier()
)

acc_val = cross_validate(pipe, data, pd.Categorical(labels).codes, cv=cv, return_estimator =True)
acc = cross_val_score(pipe, data, pd.Categorical(labels).codes, cv=cv)
mae = cross_val_score(pipe, data, pd.Categorical(labels).codes, cv=cv, 
                      scoring='neg_mean_absolute_error')


model = keras.Sequential()

model.add(layers.Dense(100, activation="relu", kernel_initializer='he_normal', bias_initializer='zeros', input_shape=data[1].shape))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(50, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(25, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(len(labels.unique()), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(data, pd.Categorical(labels).codes, test_size=0.2, shuffle=True, random_state=42)

fit = model.fit(X_train, y_train, epochs=300, batch_size=20, validation_split=0.2)

score, acc = model.evaluate(X_test, y_test,
                            batch_size=2)

print('Results - random forest')

print('Accuracy = {}, MAE = {}, Chance = {}'.format(np.round(np.mean(acc), 3), 
                                                    np.round(np.mean(-mae), 3), 
                                                    np.round(1/len(labels.unique()), 3)))

print('Results - ANN')

print('Test score:', score)
print('Test accuracy:', acc)