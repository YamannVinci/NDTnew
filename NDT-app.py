import pandas as pd
import streamlit as st
from PIL import Image
import pickle

image = Image.open('Flowchart-1.png')

st.image(image, use_column_width=True)

st.write("""
# XGBoost Prediction of Compressive Strength of Concrete Obtained from NDT Testing
This app predicts the Compressive Strength of Concrete
***
""")

# Loads the FRP-RC columns Dataset
csc = pd.read_excel("NDT.xlsx", usecols="A:AA", header=0)

# Convert data
csc['RH'] = csc['RH'].astype(float)
csc['UPV'] = csc['UPV'].astype(float)
csc['CS'] = csc['CS'].astype(float)

csc = csc[['RH', 'UPV', 'CS']]
y = csc['CS'].copy()
X = frprccolumns.drop('CS', axis=1).copy()

X_encoded = pd.get_dummies(X, columns=['Circular',
                                       'TypeCon',
                                       'TypeL',
                                       'TypeH',
                                       'Config'], drop_first=True)

# Header of Input Parameters
st.sidebar.header('Input Parameters')
value = ("0", "1")
options = list(range(len(value)))


def input_variable():
    RH = st.sidebar.slider('Rebound Number', float(X_encoded.RH.min()), float(X_encoded.RH.max()),
                               float(X_encoded.RH.mean()))
   UPV = st.sidebar.slider('UPV (km/sec)', float(X_encoded.UPV.min()), float(X_encoded.UPV.max()),
                           float(X_encoded.UPV.mean()))
  
    data = {'RH': RH,
            'UPV': UPV
            }

    features = pd.DataFrame(data, index=[0])
    return features

df = input_variable()

st.header('Specified Input Parameters')
st.write(df)
st.write('---')

xgb_model = pickle.load(open('NDT.pkl', 'rb'))

prediction = xgb_model.predict(df)[0]

st.header('Predicted Load-Carrying Capacity')
st.write('fc =', prediction, 'MPa')
st.write('---')

# Explaining the model's predictions using SHAP values
st.header('Interpretation of XGBoost prediction model using SHAP values')

image = Image.open('SHAP_summary_plot.png')
st.image(image, use_column_width=True)
st.markdown('## **SHAP summary plot**')

image = Image.open('SHAP_relative_importance.png')
st.image(image, use_column_width=True)
st.markdown('## **Relative importance for each feature**')
