
import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Reading the file
# Importing data as list ( sentences list an labels list )
# Creating a stopword array

df = pd.read_csv('bbc-text.csv')
sentences = list(df['text'])
labels = list(df['category'])
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

# Setting all hyperparameters
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type='post'
oov_tok = "<oov>"
num_epochs = 30
test_size = .2

# Splitting into train and test
train_sentences,validation_sentences,train_labels,validation_labels= train_test_split(sentences,labels,test_size=test_size,random_state=123)
print(len(train_sentences))
print(len(train_labels))
print(len(validation_sentences))
print(len(validation_labels))

# Applying tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

# Converting trained data into sequences
# Applying padding on the sequences
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding=padding_type,maxlen=max_length)

# Converting test data into sequences
# Applying padding on the sequences
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences,padding=padding_type, maxlen=max_length)

# Creating final labels for training and testing using tokenizer
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
train_labels_final = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_labels_final = np.array(label_tokenizer.texts_to_sequences(validation_labels))

# Looking at the training data. we can see how a sentence is sent for training after transforming it
reverse_word_index = dict([(value , key) for (key, value) in word_index.items()])
def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Decoded sentence for training
print(decode_sentence(train_padded[0]))
# Actual sentence
print(train_sentences[0])

# Developing model for training
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.summary()

# Training the model
history = model.fit(train_padded, train_labels_final, epochs=num_epochs, validation_data=(validation_padded, validation_labels_final))
# model.save("BBC_model.h5")
#
# # loading the saved model
# model = keras.models.load_model('BBC_model.h5')

# Plotting accuracy and loss
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# Taking a sentence and trying to predict which category it belongs to
sentence = ["hobbit picture  four years away  lord rings director peter jackson said will four years starts work film version hobbit.  oscar winner said visit sydney  desire  make  not lengthy negotiations.  think s gonna lot lawyers sitting room trying thrash deal will ever happen   said new zealander. rights jrr tolkien s book split two major film studios. jackson  currently filming remake hollywood classic king kong  said thought sale mgm studios sony corporation cast uncertainty project. 43-year-old australian city visit lord rings exhibition  attracted 140 000 visitors since opened december.  film-maker recently sued film company new line cinema undisclosed damages alleged withheld profits lost revenue first part middle earth trilogy. fellowship ring 2001 went make worldwide profits $291 million (£152 million). jackson thought secured lucrative film directing deal history remake king kong  currently production wellington. picture  stars naomi watts oscar winner adrien brody  due released december. jackson also committed making film version lovely bones  based best-selling book alice sebold."]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences,  padding=padding_type,maxlen=max_length)
print(model.predict(padded))

