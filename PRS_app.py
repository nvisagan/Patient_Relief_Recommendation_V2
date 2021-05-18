import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import joblib
from PIL import Image

st.set_page_config(page_title="MM Cabinet", page_icon="🍀",layout="wide")


logo = Image.open("MM_Logo.png")
st.image(logo,width=200)


#Write to Sidebar
st.sidebar.title('MM Cabinet Version 1.0')

st.sidebar.title('What to do')
st.sidebar.write("In the green text box on the right, Simply tell us what you are feeling or what you would like feel. You can also ask for a certain flavor or smell")
with st.sidebar.beta_expander("Expand to learn more about this project 👀"):
    st.write("As marijuana becomes widely accepted, its medical usage will rise. Today there really isn't a easy way of looking through overwhleming amounts of strains to find something that can help alleviate pain or other problems.\
        This is where MM Cabinet comes in. Out of the 2400 strains, I wanted a program that can quickly tell me what I need without complicated dropdowns or verbage.")

with st.sidebar.beta_expander("Tech Stack 💻"):
    st.write("Python 🐍",",[Spacy for NLP](https://spacy.io/)",",[NLTK for Tokenization](https://www.nltk.org/)", ",[Streamlit for the Frontend](https://streamlit.io/)" )

with st.sidebar.beta_expander("Future Iterations 🤖"):
    st.markdown("1.NLP Tuning \
    2.Corresponding Strain Images \
    3.Amazon of Medical Marijuana📦")


st.sidebar.header("Author: Navo V.")
eegg = st.sidebar.checkbox("Don't click me?")
if eegg:
    st.balloons()


#title 
st.title('MM Cabinet')
st.subheader("A patient relief recommendation system, to help users find the best strains for their problems.")


st.subheader("Tell us what you are feeling")
text = st.text_input("Ask us in plain english like the example below and we'll show what's best for you", value="I have a headache, I would like something fruity to help" ) #Text Stored as a variable


#Display results of the NLP task
st.header("MM Cabinet recommends these strains🍀:")

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
st.success('Done. Above is our top 5')