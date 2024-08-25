from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.optimizers import Adam

from Stopwords import stopwords
from bs4 import BeautifulSoup
import string
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_loss_acc(loss, accuracy):
    epochs = range(1, len(loss) + 1)

    # Wykres strat
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Wykres dokładności
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'r', label='Training accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig("Plots/Loss_and_validation")

def save_tokenizer(tokenizer):
    tokenizer_json = tokenizer.to_json()
    with open('Models/tokenizer_creating_text.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

def load_txt(path="data/Wiersze.txt"):
    sentences = ""
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sentences += str(line)
            
    f.close()

    return sentences

def clean_txt(path='data/Wiersze.txt'):
    sentences = []
    table = str.maketrans('', '', string.punctuation)
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line == '\n':
                continue
            line = line.lower()
            line = line.translate(table)  # Usuń interpunkcję
            soup = BeautifulSoup(line, 'html.parser')
            line = soup.get_text()
            line = line.replace(".", " . ")
            line = line.replace(",", " , ")
            line = line.replace("-", " - ")
            line = line.replace("/", " / ")
            words = line.split()
            
            filtered_sentence = " ".join(word for word in words if word not in stopwords)
            # filtered_sentence = " ".join(word for word in words)

            sentences.append(filtered_sentence)
    
    return sentences

def windowing(data, max_len):
    sentences = []
    words = data.split(" ")
    window_size = 10
    range_size = len(words) - max_len

    for i in range(0, range_size):
        this_sentence = ""
        for word in range(0, window_size - 1):
            word = words[i+word]
            this_sentence += word + " "
        sentences.append(this_sentence.lower())

    return sentences

if __name__ == "__main__":
    tokenizer = Tokenizer()

    data =  clean_txt()
    max_len = max([len(x) for x in data])
    # corpus = windowing(data, max_len)
    
    # for item in corpus:
    #     print(item)


    tokenizer.fit_on_texts(data)
    save_tokenizer(tokenizer)

    total_words = len(tokenizer.word_index) + 1
    print(f"\n\n\tTotal words {total_words}\n\n")



    #Turning a sequences into n_gram_sequences: [4, 2] -> [4, 2, 66] -> [4, 2, 66, 8]...
    input_sequences = []
    for line in data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    print(input_sequences[:5])

    #Save max length
    # max_len = max([len(x) for x in input_sequences])
    with open("archive/max_len.txt", 'w') as file:
        file.write(str(max_len))
    #Padding    
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_len, padding='pre'))

    #For every sequence, last item is label to be predicted
    #All rows and not last column in xs, All rows and only last column in labels
    xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
    ys = to_categorical(labels, num_classes=total_words)


    model = Sequential([
        Embedding(total_words,8),
        Bidirectional(LSTM(max_len - 1, return_sequences='True')),
        Bidirectional(LSTM(max_len - 1)),
        Dense(total_words, activation='softmax')
    ])

    Adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy', 'Precision','Recall'])

    history = model.fit(xs, ys, epochs=100, verbose=1)
    model.save(filepath='Models/Text_creator.h5')




    history_dict = history.history
    loss = history_dict['loss']
    accuracy = history_dict['accuracy']
    # plot_loss_acc(loss, accuracy)