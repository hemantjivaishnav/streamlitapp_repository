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
    
st.title(':blue[DATA ANALYSIS]')    
with st.spinner('Please Wait'):
    time.sleep(1)
    
    
allentries = os.listdir(os.getcwd()+"/uploaded")
st.subheader('Select your Record which you want to Analyse:-')
select_box = st.selectbox('',allentries)
df = pd.read_csv(select_box)
df
result = df.select_dtypes(include='number')

st.subheader('Pair Plot:-')

sns.pairplot(data = result,kind = "scatter")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


selected_cols = {}
keys = []
values = []
predict_col = {}
predict_col_list = []
count = 0
predict_user_column = ""

st.subheader("Select What Column which You Want to Predict from Dataset:-")  
for col in result.columns:
    count = count + 1
    predict_col[col] = st.checkbox(col , key = count)



for v in predict_col.keys():
    le = len(predict_user_column)
    if le == 0: 
        if predict_col[v]:
            predict_user_column = v
        else: continue
    else: continue
    

        

st.subheader("Select What Columns You Want to Drop from Training Data Set")
for cols in result.columns:
    count = count + 1
    if cols == predict_user_column:
        continue
    selected_cols[cols] = st.checkbox(cols , key = count)

for key in selected_cols.keys():
    if selected_cols[key]:
        keys.append(key)
    
   

df_columns = result.columns.tolist()
keys_diff = list(set(df_columns) - set(keys))
keys_diff.sort()

pred = st.button('predict')
if pred:
   
    train = result.drop(keys , axis=1)
    
    if not predict_user_column:
        st.warning("Please Select Column to Predict")
        st.stop()
        
    test = result[predict_user_column]

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
        
    predict_btn = st.button('predict')

    if predict_btn:
        for u in user_input.keys():
            if user_input[u] == "":
                st.warning('This is a warning', icon="⚠️")
                break
            
            else:   
                user_input = pd.DataFrame(user_input,index=[0])
                new_predict = reg.predict(user_input)

                st.subheader("Predicted House Price:")
                st.text_input("Predicted House Price",new_predict)
                st.balloons()
                st.stop()

            
    
  
    
   
        
    
    
    
   
    

        