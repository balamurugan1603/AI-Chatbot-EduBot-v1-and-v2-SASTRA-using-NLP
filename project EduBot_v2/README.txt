**************************************************************************
EduBot_V2 - AI Chat bot using intent matching - NLP algorithms and DNN
**************************************************************************
UPDATES FROM EduBot_v1:

* EduBot_v2 was trained on both monograms and bigrams.
* This was trained with Tf-Idf scores vectors of the data instead of simple word 
presence vectors which was used to train EduBot_v1.
* EduBot_v2 was trained with relatively more data compared to EduBot_v1.

**************************************************************************
FILE DESCRIPTION :

* The project folder "project EduBot_v2" contains the following files,
* EduBot_v2.rar - compressed model files (extract it and place it as child folder to the project folder).
* EduBot_v2 training script.py - Python script by which the network was trained and saved.
* EduBot_v2_main.py - Python script which runs the Chatbot.
* training_data_EduBot_v2.csv - csv file on which the network was trained.

**************************************************************************
TRAINING THE NETWORK:

* Csv file containing the questions and their intents was created.
* The csv file was loaded in the training script.
* Every words in each question was turned to lower case.
* The data was turned into sparse matrix containing Tf-Idf scores of words in each question.
* The sparse matrix was converted into array format to be fed to the network for training.
* Intents were one hot encoded.
* The network was created and trained with Tf-Idf score array and one hot encoded 
intents array. 
* The model was saved.

**************************************************************************
MODEL ARCHITECTURE:

* The network has input_shape = len(training_data_tfidf[0]).
* The first dense layer contains 10 nodes.
* The second and third dense layers contains 8 nodes each.
* The fourth dense layer contains 6 nodes.
* The output dense layer have number of nodes = len(training_data_tags_dummy_encoded[0])
and "softmax" as activation function.
* The network was compiled with "rmsprop" as optimizer and "categorical_crossentropy" as 
loss function.

**************************************************************************
FUNCTIONING OF CHATBOT:

* Csv file containing training data was loaded into EduBot_v2_main.py sript.
* Responses for each intent was created and stored as JSON file which was loaded into 
EduBot_v2_main.py script as responses dictionary.
* Every words in each question from csv data was turned to lower case.
* Tf-Idf Vectorizer for both monograms and bigrams was fit to the data.
* Label encoder object was fit to the intents.
* "predict_tag" function was created within which the input string will be transformed to
Tf-idf score vector representation and fed as input to the network and intent will be
predicted by inverse transforming the encoded labels obtained from the predicted 
probablities returned by the network. The predicted intent will be returned.
* "start_chat" function was created within which a loop is defined where input will be taken 
from the user and predicted intent will be obtained by calling "predict_tag" function on 
the user input. The predicted intent will be matched with the intents in responses dictionary
and the corresponding response is diplayed to the user.
* "start_chat" function is invoked. So that the user can start interacting with the chatbot.

**************************************************************************



