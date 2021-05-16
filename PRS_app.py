import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import joblib 

st.set_page_config(layout="wide")

#title 
st.title('Patient Relief Recommendation System')
"""
## A medical marijuana recommendation system for what you need
#### Author: **Navo V.**

"""

st.subheader("Tell us what you are feeling")
text = st.text_input('What are you feeling?', ) #Text Stored as a variable

#Display results of the NLP task
st.header("I recommend these strains")

#Data
df = pd.read_csv("https://raw.githubusercontent.com/build-week-med-cabinet-2/ML_Model-Data/master/Cannabis_Strains_Features.csv")


#Load in pickles
nn = joblib.load('NN_MJrec.pkl')
tfidf = joblib.load('tfidf.pkl')

def recommend(text):
   # Transform
    text = pd.Series(text)
    vect = tfidf.transform(text)

    # Send to df
    vectdf = pd.DataFrame(vect.todense())
    

    # Return a list of indexes
    top5 = nn.kneighbors([vectdf][0], n_neighbors=5)[1][0].tolist()
   
    
    # Send recomendations to DataFrame
    recommendations_df = df.iloc[top5]
    recommendations_df['index']= recommendations_df.index
    
    #return recommendations_df

    st.dataframe(recommendations_df, width=7000, height=768)


recommend(text)