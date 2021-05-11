import nltk
import json
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model

# loading training data from json file
with open("training_data_EduBot_v1.json", "r") as file:
    intents = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

# extracting words and labels(tags) from the data
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        docs_x.append(tokens)
        docs_y.append(intent["tag"])
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# stemming tokenized words
stemmer = LancasterStemmer()
words = sorted(list(set([stemmer.stem(w.lower()) for w in words if w != "?"])))
labels = sorted(labels)

# creating bag of words and corresponding labels
training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    tokens = [stemmer.stem(w.lower()) for w in doc]
    for w in words:
        if w in tokens:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

# converting bag of words and their corresponding labels into numpy arrays
training = np.array(training)
output = np.array(output)

# creating DNN
edubot = Sequential()
edubot.add(Dense(8, input_shape=(len(training[0]),)))
edubot.add(Dense(8))
edubot.add(Dense(len(output[0]), activation="softmax"))
edubot.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

edubot.fit(training, output, epochs=118, batch_size=20)

# saving the model
save_model(edubot, "EduBot_v1")