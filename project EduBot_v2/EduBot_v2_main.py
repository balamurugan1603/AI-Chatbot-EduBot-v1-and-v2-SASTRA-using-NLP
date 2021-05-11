# importing modules
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.models import load_model
import json
import random


# importing training data
training_data = pd.read_csv("training_data_EduBot_v2.csv")

# loading model
edubot_v2 = load_model("EduBot_v2")

# importing responses
responses = json.load(open("responses.json", "r"))


# fitting TfIdfVectorizer with training data to preprocess inputs
training_data["patterns"] = training_data["patterns"].str.lower()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
vectorizer.fit(training_data["patterns"])


# fitting LabelEncoder with target variable(tags) for inverse transformation of predictions
le = LabelEncoder()
le.fit(training_data["tags"])


# preprocessing input
def predict_tag(inp_str):
    inp_data_tfidf = vectorizer.transform([inp_str.lower()]).toarray()
    predicted_proba = edubot_v2.predict(inp_data_tfidf)
    encoded_label = [np.argmax(predicted_proba)]
    predicted_tag = le.inverse_transform(encoded_label)[0]
    return predicted_tag


# chat function
def start_chat():
    print("---------------  EduBot_v2 - AI Chat bot  ---------------")
    print("Ask any queries regarding SASTRA...")
    print("I will try to understand you and reply...")
    print("Type EXIT to quit...")
    while True:
        inp = input("Ask anything... : ")
        if inp == "EXIT":
            break
        else:
            if inp:
                tag = predict_tag(inp)
                response = random.choice(responses[tag])
                print("Response... : ", response)
            else:
                pass

# calling chat function to check functionality of the program
start_chat()