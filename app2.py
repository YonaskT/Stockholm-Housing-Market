import joblib
import pandas as pd 
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df_sthlm=pd.read_csv('result_ml.csv')
df_sthlm.drop(columns=['Address','Location','District','lat_2', 'long_2','Latitude','Longitude'],axis=1,inplace=True)
categorical_cols = ['Balcony', 'Patio', 'Elevator']
df_sthlm[categorical_cols] = df_sthlm[categorical_cols].apply(lambda col: col.astype('category').cat.codes)
le=LabelEncoder()
df_sthlm['Municipality']=le.fit_transform(df_sthlm['Municipality'])
df_sthlm.drop(columns=['Monthly Pay','Patio'],axis=1,inplace=True)
X=df_sthlm.drop(['Price','Elevator'],axis=1)
y=df_sthlm['Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
# Example training data
# X_train, y_train = your_training_data()

# Fit the scaler on training data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)

# Train the model
rf_top = RandomForestRegressor(n_estimators=10, min_samples_split=2, min_samples_leaf=2, max_depth=None, random_state=1)
rf_top.fit(x_train_scaled, y_train)

# Save the scaler and model
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(rf_top, 'rf_top.joblib')