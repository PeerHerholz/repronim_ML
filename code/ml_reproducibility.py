import json
import os
from pathlib import Path
from joblib import dump

import random
from re import A
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_score
from joblib import dump

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

repo_path = Path(__file__).parent.parent

data = np.load(repo_path / 'data/raw/a.npy')
data.shape

labels = pd.read_csv(repo_path /  'data/raw/participants.csv')['AgeGroup']

cv = StratifiedKFold(random_state=42, shuffle=True)

pipe = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

acc_val = cross_validate(pipe, data, pd.Categorical(labels).codes, cv=cv, return_estimator =True)
acc = cross_val_score(pipe, data, pd.Categorical(labels).codes, cv=cv)
mae = cross_val_score(pipe, data, pd.Categorical(labels).codes, cv=cv, 
                      scoring='neg_mean_absolute_error')

os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)

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

tf.random.set_seed(42)

fit = model.fit(X_train, y_train, epochs=300, batch_size=20, validation_split=0.2)

score, acc_ann = model.evaluate(X_test, y_test,
                            batch_size=2)

print('Results - random forest')

print('Accuracy = {}, MAE = {}, Chance = {}'.format(np.round(np.mean(acc), 3), 
                                                    np.round(np.mean(-mae), 3), 
                                                    np.round(1/len(labels.unique()), 3)))

print('Results - ANN')

print('Test score:', score)
print('Test accuracy:', acc_ann)

metrics = {"accuracy": np.round(np.mean(acc), 3), "MAE": np.round(np.mean(-mae), 3), "Chance": np.round(1/len(labels.unique()), 3),
           "Test score": score, "Test accuracy": acc_ann}

accuracy_path = repo_path / "metrics.json"
accuracy_path.write_text(json.dumps(metrics))

dump(acc_val, repo_path / "random_forest.joblib")

model.save(repo_path / "ANN.h5")