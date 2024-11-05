import streamlit as st
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from bpemb import BPEmb
from sklearn.feature_extraction.text import CountVectorizer


bpemb = BPEmb(lang='en')

def bpemb_tokenizer(text):
    return bpemb.encode(text)


def bpemb_tokenizer(text):
    return bpemb.encode(text)


def load_model(filename):
  with open(filename, 'rb') as file:
    model = pickle.load(file)

  return model

def load_cv():
  with open('countvector.pkl', 'rb') as file:
    cv = pickle.load(file)
  return cv

def get_example():
  example = pd.read_csv('example.csv')
  return example

nltk.download('stopwords')
def process(text):
  text = text.replace('%', ' percent')
  text = text.replace('$', ' dollar ')
  text = text.replace('₹', ' rupee ')
  text = text.replace('€', ' euro ')
  text = text.replace('@', ' at ')
  prunct = [",", "?" , "." , ";" , ":" , "!" , '"', "ๆ" , "ฯ"]
  clean = [x.lower() for x in text if x not in prunct]
  clean = ''.join(clean)

  stop_words =set(stopwords.words('english'))
  clean = [word for word in clean.split() if word.lower() not in stop_words]

  final = ' '.join(clean)
  return final

example = get_example()
cnnModel = load_model("cnn.pkl")
maxEntModel = load_model("maxEnt.pkl")
naiveBayesModel = load_model("naiveBayes.pkl")
xgboostModel = load_model("xgb.pkl")
cv = load_cv()

st.set_page_config(layout="wide")

exampleContainer, mainContainer = st.columns([0.5, 0.5], gap="medium")

with exampleContainer:
  st.title(":red[Example Questions]")
  st.write("In the column is_duplicate, 1 means the questions are same meaning and 0 means they are not.")
  st.write(example.drop(columns=['id', 'qid1', 'qid2']))

with mainContainer:
  question1 = st.text_input('Enter your question 1 here:', key="q1", placeholder="Question 1")
  question2 = st.text_input('Enter your question 2 here:', key="q2", placeholder="Question 2")
  if st.button('Predict'):
    st.write(question1)
    st.write(question2)
    if question1 and question2:
      question1 = process(question1)
      question2 = process(question2)
      questions = cv.transform([question1, question2])

      cnnPredict = cnnModel.predict(questions)
      maxEntPredict = maxEntModel.predict(questions)
      naiveBayesPredict = naiveBayesModel.predict(questions)
      xgboostPredict = xgboostModel.predict(questions)
      
      cnnPredcitC = st.container()
      maxEntPredictC = st.container()
      naiveBayesPredictC = st.container()
      xgboostPredictC = st.container()
      
      with cnnPredcitC:
        c1, c2 = st.columns([0.5, 0.5])
        with c1:  
          st.subheader("CNN Prediction")
          st.write(cnnPredict)

      with maxEntPredictC:
        c1, c2 = st.columns([0.5, 0.5])
        with c1:  
          st.subheader("Max Entropy Prediction")
          st.write(maxEntPredict)
          
      with naiveBayesPredictC:
        c1, c2 = st.columns([0.5, 0.5])
        with c1:  
          st.subheader("Naive Bayes Prediction")
          st.write(naiveBayesPredict)
          
      with xgboostPredictC:
        c1, c2 = st.columns([0.5, 0.5])
        with c1:  
          st.subheader("XGBoost Prediction")
          st.write(xgboostPredict)

    else:
      st.write("Please enter questions to predict")