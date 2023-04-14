import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import time

with st.spinner('Please Wait'):
    time.sleep(5)
allentries = os.listdir(os.getcwd()+"/uploaded")
select_box = st.selectbox('Select your Record which you want to Analyse:-',allentries)
df = pd.read_csv(select_box)
df
result = df.select_dtypes(include='number')

sns.pairplot(data = result,kind = "scatter")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


selected_cols = {}
keys = []
values = []



st.subheader("Select What Columns You Want to Drop from Training Data Set")
for cols in result.columns:
    if cols=="price":
        continue
    selected_cols[cols] = st.checkbox(cols)


for key in selected_cols.keys():
    if selected_cols[key]==True:
        keys.append(key)
    
   
keys.append("price")

df_columns = result.columns.tolist()
keys_diff = list(set(df_columns) - set(keys))
keys_diff.sort()


train = result.drop(keys , axis=1)
test = result['price']

x_train , x_test , y_train , y_test = train_test_split(train , test , test_size=0.2 , random_state = 2)

reg = LinearRegression()
reg.fit(x_train , y_train)

predicted_data = reg.predict(x_test)

plt.scatter(y_test,predicted_data)
st.pyplot()

reg_score = reg.score(x_test , y_test)

title = st.text_input('Regression Score', reg_score)
  

st.subheader("Please give your own Input to Predict House Price")

user_input = {}

for key in keys_diff:
    user_input[key] = st.text_input(key)

user_input = pd.DataFrame(user_input,index=[0])
new_predict = reg.predict(user_input)

st.subheader("Predicted House Price:")
st.text_input("Predicted House Price",new_predict)
        
    
  
    
   
        
    
    
    
   
    

        