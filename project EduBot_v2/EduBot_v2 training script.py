# importing modules
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model


# importing training data
training_data = pd.read_csv("training_data_EduBot_v2.csv")

# preprocessing training data
training_data["patterns"] = training_data["patterns"].str.lower()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
training_data_tfidf = vectorizer.fit_transform(training_data["patterns"]).toarray()

# preprocessing target variable(tags)
le = LabelEncoder()
training_data_tags_le = pd.DataFrame({"tags": le.fit_transform(training_data["tags"])})
training_data_tags_dummy_encoded = pd.get_dummies(training_data_tags_le["tags"]).to_numpy()

# creating DNN
edubot = Sequential()
edubot.add(Dense(10, input_shape=(len(training_data_tfidf[0]),)))
edubot.add(Dense(8))
edubot.add(Dense(8))
edubot.add(Dense(6))
edubot.add(Dense(len(training_data_tags_dummy_encoded[0]), activation="softmax"))
edubot.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# fitting DNN
edubot.fit(training_data_tfidf, training_data_tags_dummy_encoded, epochs=50, batch_size=32)

# saving model file
save_model(edubot, "EduBot_v2")