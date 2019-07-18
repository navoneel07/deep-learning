import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from pickle import dump

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

in_filename = "kafka_sequences.txt"
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

chars = sorted(list(set(raw_text)))
char_map = {c:i for i,c in enumerate(chars)}

sequences = []
for line in lines:
    encoded_seq = [char_map[char] for char in line]
    sequences.append(encoded_seq)

vocab_size = len(char_map)
sequences = np.array(sequences)

X, y = sequences[:, :-1], sequences[:, -1]

sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = np.array(sequences)
y = to_categorical(y, num_classes=vocab_size)

#define model

'''model = Sequential()
model.add(LSTM(128, input_shape = (X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation = 'softmax'))

#compile the model
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

model.fit(X, y, epochs = 100)

model.save('rnn_character.model')'''

dump(char_map, open('mapping.pickle', 'wb'))
