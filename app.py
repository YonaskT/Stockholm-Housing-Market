import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.title('Stockholm Housing Price Prediction')
st.markdown('''
This is a Streamlit web app that can be used to predict apartment prices in Stockholm, Sweden.
The app uses machine learning to predict the price of the house. 
It loads a pre-trained random forest regression model, which takes as input various features of the house, 
such as size of the house, the number of rooms, the amount per sqm, the floor, and the distance to the city center. ''')
st.markdown('The dataset for this project is created by scraping data from Sweden largest housing listing sites [Booli](https://www.booli.se/) and [Hemnet](https://www.hemnet.se/). ')
df_sthlm=pd.read_csv('result_ml.csv',usecols=lambda x:x not in ['lat_2','long_2'])

#Show Housing Data
#if st.checkbox('Show Apartmnet Data'):
st.subheader('Apartment Data')
st.write(df_sthlm.head())
if st.checkbox('Show Descriptive Statistics'):
    st.subheader('Descriptive Statistics')
    st.write(df_sthlm.describe())

import matplotlib.pyplot as plt
import seaborn as sns


# Define your numeric features
numeric_features = df_sthlm.select_dtypes('number')
# Create the plot
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 14))
axes = axes.flatten()

for i, column in enumerate(numeric_features):
    if i >= len(axes):
        break
    mean_val = df_sthlm[column].mean()
    median_val = df_sthlm[column].median()
    axes[i].axvline(mean_val, color='black', linestyle='--', label=f'Mean = {mean_val:.2f}')
    axes[i].axvline(median_val, color='green', linestyle='--', label=f'Median = {median_val:.2f}')
    axes[i].legend()
    sns.histplot(data=df_sthlm, x=column, ax=axes[i], kde=True, color='blue')
    axes[i].set_title(column, fontfamily='serif')

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()

# Display the plot in Streamlit
if st.checkbox('Show Distribution of the dataset'):
   st.subheader('Hist Plots')
   st.pyplot(fig)

# Load the trained model and scaler
model = joblib.load('rf_top.joblib')
scaler = joblib.load('scaler.joblib')

if not model:
    st.error("Model failed to load.")
if not scaler:
    st.error("Scaler failed to load.")
#Cols = ['Size in m²', 'Total Rooms', 'Floor', 'Amt per m²', 'Balcony','Municipality', 'Distance']
Cols= ['Amt per m²', 'Balcony', 'Floor', 'Municipality',
       'Size in m²', 'Total Rooms', 'Distance']
municipalities = [
    'Stockholms kommun', 'Järfälla kommun', 'Huddinge kommun', 'Nacka kommun', 'Sollentuna kommun',
    'Norrtälje kommun', 'Haninge kommun', 'Värmdö kommun', 'Sundbybergs kommun', 'Upplands-Bro kommun',
    'Upplands Väsby kommun', 'Södertälje kommun', 'Botkyrka kommun', 'Tyresö kommun', 'Sigtuna kommun',
    'Solna kommun', 'Täby kommun', 'Österåkers kommun', 'Salems kommun', 'Vallentuna kommun', 'Danderyds kommun',
    'Vaxholms kommun', 'Lidingö kommun', 'Nynäshamns kommun', 'Nykvarns kommun', 'Ekerö kommun'
]
le = LabelEncoder()
le.fit(municipalities)

# Streamlit UI for inputs
Municipality = st.selectbox('Municipality', municipalities)
Area = st.number_input("Area", 15, 235)
Rooms = st.number_input("Number of rooms", 1, 8)
Floor = st.number_input('Floor', -2, 29)
AmountSqm = st.number_input('Amt per m²', 12097, 185536)
Distance = st.number_input('Distance', 0, 46)
Balcony = st.radio("Balcony", ["Yes", "No"]) == "Yes"


if st.button('Predict'):
    output = {
        
        'Size in m²': [Area],
        'Total Rooms': [Rooms],
        'Floor': [Floor],
        'Amt per m²':  [AmountSqm], 
        'Balcony': [Balcony],
        'Municipality': [le.transform([Municipality])[0]],        
        'Distance': [Distance]     
              
    }

    x = pd.DataFrame(output)

    # Check for missing values
    if x.isnull().values.any():
        st.error("Input data contains missing values. Please fill all fields.")
    else:
        # Ensure the columns match those used during training
        x = x[Cols]

        # Check the input data
        st.write("Input Data:", x)

        try:
            # Transform the input data using the loaded scaler
            x_scaled = scaler.transform(x)

            # Check the scaled data
            st.write("Scaled Data:", x_scaled)

            # Predict using the loaded model
            predictionX = model.predict(x_scaled)
            st.success(f'The price of this house is {predictionX[0]:,.2f}!')
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")