# importing modules
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import random
import json
from tensorflow.keras.models import load_model

# loading training data
with open("training_data_EduBot_v1.json", "r") as file:
    intents = json.load(file)

words = []
labels = []

# preparing words array and labels
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# stemming words
stemmer = LancasterStemmer()
words = sorted(list(set([stemmer.stem(w.lower()) for w in words if w != "?"])))
labels = sorted(labels)


# function to create bag of words from input
def bow(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(w.lower()) for w in s_words]
    for se in s_words:
        for i, w in enumerate(s_words):
            if w == se:
                bag[i] = 1
    return np.array(bag).reshape(-1, 68)


# loading model
edubot_v1 = load_model("EduBot_v1")


# function to predict tags from input string
def predict_tag(inp):
    inp_bow = bow(inp, words)
    predicted_proba = edubot_v1.predict(inp_bow)
    index = np.argmax(predicted_proba)
    predicted_tag = labels[index]
    return predicted_tag


# function to initiate chat
def start_chat():
    print("---------------  EduBot_v1 - AI Chat bot  ---------------")
    print("Ask any queries regarding SASTRA...")
    print("I will try to understand you and reply...")
    print("Type EXIT to quit...")
    while True:
        inp_str = input("Ask anything... : ")
        if inp_str == "EXIT":
            break
        else:
            if inp_str:
                for intent in intents["intents"]:
                    if intent["tag"] == predict_tag(inp_str):
                        response = random.choice(intent["responses"])
                        print("Response... : ", response)
            else:
                pass

# start chatting
start_chat()