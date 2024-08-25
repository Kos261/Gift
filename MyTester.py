from keras.preprocessing.text import tokenizer_from_json
from keras.models import load_model
from keras.utils import pad_sequences
import json
import numpy as np
import os
from bs4 import BeautifulSoup
import string
from Stopwords import stopwords
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Ustawia poziom logowania na WARN



def creating_text(next_words, tokenizer, model, seed_text):
    print(f"\n\n\tSEED TEXT: {seed_text}")
    word_index = tokenizer.word_index
    reversed_word_index = {value: key for key, value in word_index.items()}

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list],maxlen=max_len, padding='pre')

        predicted = np.argmax(model.predict(token_list), axis=-1)[0]
        predicted_word = reversed_word_index[predicted]

        seed_text += " " + predicted_word

    print(f"\n\n\tPREDICTED TEXT: {seed_text}")



if __name__ == "__main__":
    # Wczytanie tokenizera z pliku JSON
    with open('Models/tokenizer_creating_text.json', 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)



    # Wczytywanie max_len
    with open('archive/max_len.txt', 'r') as f:
        lines = f.readlines()
        max_len = int(lines[0])

    model = load_model(filepath='Models/Text_creator.h5')


   
    ###Creating text
    seed_text = "Miłość jest"
    creating_text(tokenizer=tokenizer, next_words=50, model=model, seed_text=seed_text)
