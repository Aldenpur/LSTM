import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re
import glob
import nltk
import string
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.preprocessing import sequence
from keras.optimizers import Adam
import numpy as np


folder_path = "Dataset/academic_plagiarism"
folder_path2 = "Dataset/src/Central1"
#folder_path = "Dataset/plagiarism1"
#folder_path2 = "Dataset/source1"

file_list = glob.glob(folder_path + "/*.txt")
file_list2 = glob.glob(folder_path2 + "/*.txt")

text_list = []

for file_path in file_list:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        text_list.append([text, 0])

for file_path2 in file_list2:
    with open(file_path2, "r", encoding="utf-8") as file2:
        text = file2.read()
        text_list.append([text, 1])

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('popular')


def preprocess_documents(documents, max_vocab_size, max_sequence_length):
    stop_words = set(stopwords.words('russian'))
    documents_cleaned = []
    for document in documents:
        text = document[0].translate(str.maketrans("", "", string.punctuation))
        text = [word.lower() for word in text.split() if word.lower() not in stop_words]
        text = " ".join(text)
        identifier = document[1]
        documents_cleaned.append((text, identifier))


    tokenizer = Tokenizer(num_words=max_vocab_size)
    print('Stage 1')

    tokenizer.fit_on_texts([document[0] for document in documents_cleaned])
    print('Stage 2')

    sequences = tokenizer.texts_to_sequences([document[0] for document in documents_cleaned])
    print('Stage 3')

    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    print('Stage 4')

    preprocessed_documents = [(padded_sequences[i], documents_cleaned[i][1]) for i in range(len(padded_sequences))]
    print('Stage 5')

    return tokenizer, preprocessed_documents


max_vocab_size = 20000
max_sequence_length = 1000
embedding_dim = 200

tokenizer, preprocessed_documents = preprocess_documents(text_list, max_vocab_size, max_sequence_length)

print("Vocabulary size:", len(tokenizer.word_index))
print("Sequence length:", preprocessed_documents[0][0].shape[0])

text = [pair[0] for pair in preprocessed_documents]
ids = [pair[1] for pair in preprocessed_documents]

train_text, test_text, train_ids, test_ids = train_test_split(text, ids, test_size=0.2, random_state=42)

train_text, val_text, train_ids, val_ids = train_test_split(train_text, train_ids, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(max_vocab_size, embedding_dim, input_length=max_sequence_length))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

print(model.summary())

x_train = np.array(train_text)
y_train = np.array(train_ids)
val_text = np.array(val_text)
val_ids = np.array(val_ids)
test_text = np.array(test_text)
test_ids = np.array(test_ids)

history = model.fit(x_train, y_train, validation_data=(val_text, val_ids), batch_size=8, epochs=2, verbose=1)

loss, accuracy = model.evaluate(test_text, test_ids, verbose=1)
print('Test loss:', loss)
print('Test accuracy:', accuracy)







