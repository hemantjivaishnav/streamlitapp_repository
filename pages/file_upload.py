import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil

current_dir = os.getcwd()
file = st.file_uploader('Select file')
file
st.stop()
if file:
    if file.type=="text/csv":
        filepath = current_dir+"/"+file.name
        df = pd.read_csv(filepath)
        targetpath = r"C:\Users\Hemant Vaishnav\Documents\django\uploaded/"+file.name
        shutil.copyfile(filepath,targetpath)
        df
        st.success("File Uploaded Successfully")
    else:
        st.warning("upload csv file only!!!!!")
        
    
    
