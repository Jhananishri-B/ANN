import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)

st.title("Wine Quality Prediction 🍷")
st.write("Enter the wine properties below to predict the wine quality (3-8):")

feature_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    user_input.append(value)

input_df = pd.DataFrame([user_input], columns=feature_names)

df = pd.read_csv("winequality-red.csv")
X = df.drop("quality", axis=1)
y = df["quality"]

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
input_scaled = scaler.transform(input_df)

y_train_cat = to_categorical(y_train - y_train.min())
y_test_cat = to_categorical(y_test - y_test.min())

model = Sequential([
    Dense(256, input_dim=X_train.shape[1], activation="relu"),
    Dropout(0.4),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(y_train_cat.shape[1], activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

with st.spinner("Training the model... this may take a minute"):
    history = model.fit(
        X_train, y_train_cat,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[es],
        verbose=0
    )

y_pred = model.predict(input_scaled)
predicted_quality = np.argmax(y_pred, axis=1)[0] + y_train.min()

st.success(f"Predicted Wine Quality: {predicted_quality}")
