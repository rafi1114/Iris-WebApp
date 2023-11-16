import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# My first ML Web App Hehe
          
Predicting Iris Flower Type

""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4) #last value is default value
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('petal Width', 0.1, 2.5, 0.2)
    data = {'sepal_length' : sepal_length,
            'sepal_width' : sepal_width,
            'petal_length' : petal_length,
            'petal_width' : petal_width}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

prediction = clf.predict(df)
prediction_probability = clf.predict_proba(df)


st.subheader('Class Labels and their index numbers')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_probability)

