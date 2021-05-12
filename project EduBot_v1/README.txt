**************************************************************************
EduBot_V1 - AI Chat bot using intent matching - NLP algorithms and DNN
**************************************************************************
 FILE DESCRIPTION :

* The project folder "project EduBot_v1" contains the following files,
* EduBot_v1.rar - compressed model files (extract it and place it as child folder to the project folder).
* EduBot_v1 training script.py - Python script by which the network was trained and saved.
* EduBot_v1_main.py - Python script which runs the Chatbot.
* training_data_EduBot_v1.json - json data on which the network was trained.

**************************************************************************
TRAINING THE NETWORK:

* Json file containing intents, patterns (questions) and the responses for each intents was created.
* The json file was loaded in the training script.
* All the questions in the file were tokenized (only monograms) and added to the words array.
* Tokens in each question is grouped into an array and appended to the docs_x array 
and their corresponding intents to docs_y array.
* All the unique intents were added to the labels array.
* Each token in the words array was stemmed into its root word.
* Comparing the tokens in words array and tokens in each row of docs_x array, bag of words 
representation is created for each array in docs_x and appended to the training array and the 
output array is created with dummy variable representation of labels array.
* The network was created and trained with the training array and output array.
* The model was saved.

**************************************************************************
MODEL ARCHITECTURE:

* The Network has input_shape = len(training[0]).
* Contains one Dense layer with 8 nodes.
* Contains another Dense layer with 8 nodes.
* The output Dense layer of the network has number of nodes equal to len(output[0]) and
"softmax" as activation function.
* The network was compiled with "rmsprop" as optimizer and "categorical_crossentropy" as
loss function.

**************************************************************************
FUNCTIONING OF CHATBOT:

* Json file containing training data is loaded into EduBot_v1_main.py script.
* All the questions in the file were tokenized (only monograms) and added to the words array.
* All the unique intents were added to the labels array.
* Each token in the words array was stemmed into its root word.
* "bow" function was created within which the given input is stemmed to its root word
and then converted to bag of words format by comparing the tokens in given input with 
tokens in words array. The bag of words array of the input will be returned.
* "predict_tag" function was created within which the bag of words array of the input is 
fed to the network and predicted probablities of different intents are obtained from which 
we can find the predicted intent. The predicted intent will be returned.
* "start_chat" function is created within which a loop is defined where input is taken from the 
user and predicted intent is obtained by calling "predict_tag" function on the user input. From the
predicted intent, Response is generated from the training data by matching the intent predicted
with the intents in training data and displayed to the user.
* "start_chat" function is invoked. So that the user can start interacting with the chatbot.

**************************************************************************
NOTE: 
* EduBot_v1 is a failure version and was not learning the patterns in the data so well.
Because it was trained only on monograms and simple bag of words representation of 
data where value 1 if word is present and 0 if word is not present in the word corpus. 
* The scripts were analysed and the above reasons were detected as the reason for
this inaccuracy.
* These were optimised in EduBot_v2.

