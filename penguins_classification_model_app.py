import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


# First section focuses on collecting the parameters for prediction and converting them into a format for accurate prediction
st.image('penguin_unsplash.jpg', caption='Photo by Derek Oyen on Unsplash')
st.title('Penguin Prediction App')
st.markdown('This app helps you predict from three types of penguins(Adelie, Gentoo, or Chinstrap) based on features you supply')
st.header('Feature Selection')
st.write('**Select the features here:**')

# The two list below are the unique values of the island and sex columns in the penguins dataset
island_options = ['Biscoe','Dream','Torgersen']
sex_options = ['male', 'female']

# These assign values to the selected features and make them variables for the prediction
island = st.selectbox('**Select the Island where the Penguin is found**', island_options)
bill_length_mm = st.slider('**Select the bill length (mm) parameter**', 10.0, 75.0,32.1 )
bill_depth_mm = st.slider('**Select the bill depth (mm) parameter**', 10.0, 35.0,13.1 )
flipper_length_mm = st.slider('**Select the flipper length (mm) parameter**', 120.0, 325.0,172.0 )
body_mass_g = st.slider('**Select the body mass (grams) parameter**', 2200.0, 9200.0,2700.0 )
sex = st.selectbox('**Select the gender(sex) of the penguin**', sex_options )

# This section collects the features and puts them in a dictionary which is then converted to a dataframe
selected_dic = {'island' : island,'bill_length_mm': bill_length_mm,'bill_depth_mm' : bill_depth_mm,
                 'flipper_length_mm' : flipper_length_mm, 'body_mass_g' : body_mass_g, 'sex' : sex}

selected_features = pd.DataFrame(selected_dic,  index = [0])
st.write('**Your selected Penguin Features are as shown below**')
st.dataframe(selected_features)

# This section transforms the data elements in the dataset using label assignment for categorical values
# And StandardScalar for numerical values

# For island mapping
if island == 'Biscoe':
    island = 0 
elif island == 'Dream':
    island = 1
elif island == 'Torgersen':
    island = 2

# For sex mapping
if sex == 'male':
    sex = 1 
elif sex == 'female':
    sex = 0

selected_dic_tf = {'island' : island,'bill_length_mm': bill_length_mm,'bill_depth_mm' : bill_depth_mm,
                 'flipper_length_mm' : flipper_length_mm, 'body_mass_g' : body_mass_g, 'sex' : sex}

selected_features_tf = pd.DataFrame(selected_dic_tf, index=[0])

sc = StandardScaler()
selected_features_tf[['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']]=sc.fit_transform(selected_features_tf[['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']])

st.write('**Your selected Penguin transformed Features are as shown below**')
st.dataframe(selected_features_tf) # This is the fully transformed dataset using Categorical values mapping and Scaled Numerical values

# Loading the trained model using joblib
model = joblib.load('trained_penguinlg.joblib')

# Making prediction based on user input
if st.button('predict'):
    # This predicts the penguin specie using the prebuild model imported using joblib
    prediction = model.predict(selected_features_tf)
    # This writes the prediction
    st.write(f'Predicted Class of Penguin is : {prediction[0]}')