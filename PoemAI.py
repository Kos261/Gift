import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QTextEdit, QVBoxLayout, QComboBox, QGridLayout, QLabel, QSlider, QLineEdit
from PyQt5.QtGui import QPainter, QPixmap, QColor, QIcon, QFont
from PyQt5.QtCore import QThread, pyqtSignal, QTimer

from random import choice
import random
from Wiersze import Wiersze
from keras.preprocessing.text import tokenizer_from_json
from keras.models import load_model
from keras.utils import pad_sequences
import json
import numpy as np
import time


class StartScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.seed_list = ["Miłość jest jak", "Kocham cię jak", "Jesteś niczym", "Kiedy","Kocham"]
        self.load_model_and_tokenizer()
        self.setFixedSize(550, 550)
        self.Layout = QVBoxLayout(self)
        self.button = QPushButton("Generuj wiersz miłosny", self)
        self.button.clicked.connect(self.get_random_answer)
        self.Layout.addWidget(self.button)
        

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)  # Ustaw tryb tylko do odczytu
        self.Layout.addWidget(self.text_edit)


        self.settings = QPushButton()
        self.settings.setFixedSize(32, 32)
        self.settings.setIcon(QIcon("Images/Gear.png"))
        self.settings.clicked.connect(self.clickedSettings)
        self.Layout.addWidget(self.settings)

        self.loading_dots = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loading_dots)

    def update_loading_dots(self):
        dots = '.' * (self.loading_dots % 4)
        self.text_edit.setPlainText(f'Loading{dots}')
        self.loading_dots += 1

    def get_random_answer(self):
        self.text_edit.clear()
        # Uruchamiamy animację ładowania
        self.loading_dots = 0
        self.timer.start(500)
        choice = random.random()
        if choice > 0 and choice < 0.7:
            
            text = '\n' + str(random.choice(Wiersze))
        else:
            seed = random.choice(self.seed_list)
            text = '\n' + self.creating_text(50, self.tokenizer, self.model, seed)
        self.text_edit.append(text)
        self.timer.stop()

    def load_model_and_tokenizer(self):
          # Wczytanie tokenizera z pliku JSON
        with open('Models/tokenizer_creating_text.json', 'r', encoding='utf-8') as f:
            tokenizer_json = json.load(f)
        self.tokenizer = tokenizer_from_json(tokenizer_json)


        # Wczytywanie max_len
        with open('archive/max_len.txt', 'r') as f:
            lines = f.readlines()
            self.max_len = int(lines[0])

        self.model = load_model(filepath='Models/Text_creator.h5')

    def creating_text(self, next_words, tokenizer, model, seed_text):
        print(f"\n\n\tSEED TEXT: {seed_text}")
        word_index = tokenizer.word_index
        reversed_word_index = {value: key for key, value in word_index.items()}

        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list],maxlen=self.max_len, padding='pre')

            predicted = np.argmax(model.predict(token_list), axis=-1)[0]
            predicted_word = reversed_word_index[predicted]

            seed_text += " " + predicted_word

        print(f"\n\n\tPREDICTED TEXT: {seed_text}")
        return seed_text

    def clickedSettings(self):
        self.Settings = SettingsScreen(self, self.text_edit)
        self.Settings.show()

class SettingsScreen(QWidget):
    def __init__(self, Startscreen, text_edit) -> None:
        super().__init__()
        self.Layout = QGridLayout(self)
        
        self.text_edit = text_edit
        self.setFixedSize(300, 300)
        self.fontsize = 1
        self.label2 = QLabel(f"Rozmiar czcionki: {self.fontsize}")
        self.slider = QSlider(Qt.Horizontal)
        self.button = QPushButton("Sejf?")
        self.button.clicked.connect(self.open_vault)
        self.button.setFixedSize(90, 60)
        self.slider.setTickInterval(5)
        self.slider.setMinimum(1)
        self.slider.setMaximum(50)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.value_changed)



        self.Layout.addWidget(self.label2, 1,0)
        self.Layout.addWidget(self.slider, 2,0)
        self.Layout.addWidget(self.button, 3,0)
        
    def value_changed(self):
        self.fontsize = self.slider.value()
        self.label2.setText(f"Rozmiar czionki: {self.fontsize}")

        label_font = QFont("Arial", self.fontsize)  # Nazwa czcionki i rozmiar
        self.label2.setFont(label_font)
        self.text_edit.setFont(label_font)

    def open_vault(self):
        vault = PinWidget()
        vault.show()

class PinWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 150)
        
        # Ustawienie layoutu siatki
        self.layout = QGridLayout(self)
        
        # Lista przechowująca pola QLineEdit
        self.pin_fields = []
        
        # Tworzenie 8 pól QLineEdit
        for i in range(8):
            line_edit = QLineEdit()
            line_edit.setMaxLength(1)  # Ograniczenie do jednego znaku
            line_edit.setFixedSize(40, 40)  # Ustawienie rozmiaru pola
            line_edit.setAlignment(Qt.AlignCenter)  # Wycentrowanie tekstu
            
            # Dodanie pola do layoutu i listy
            self.layout.addWidget(line_edit, 0, i)
            self.pin_fields.append(line_edit)
            
            # Przejście do następnego pola po wpisaniu znaku
            line_edit.textChanged.connect(lambda _, i=i: self.focus_next_field(i))

        # Przycisk do weryfikacji PIN-u
        self.verify_button = QPushButton("Sprawdź PIN")
        self.verify_button.clicked.connect(self.verify_pin)
        self.layout.addWidget(self.verify_button, 1, 0, 1, 8)  # Przycisk na dole pod polami
        
        # Etykieta do wyświetlania komunikatów
        self.message_label = QLabel("")
        self.layout.addWidget(self.message_label, 2, 0, 1, 8, alignment=Qt.AlignCenter)

    def focus_next_field(self, index):
        if index < 7:  # Przejście do kolejnego pola tylko jeśli nie jesteśmy w ostatnim
            self.pin_fields[index + 1].setFocus()

    def get_pin(self):
        # Pobranie PIN-u z pól
        pin = ''.join([field.text() for field in self.pin_fields])
        return pin

    def clear_pin_fields(self):
        # Wyczyść wszystkie pola QLineEdit
        for field in self.pin_fields:
            field.clear()

    def verify_pin(self):
        # Przykład poprawnego PIN-u
        correct_pin = "30027052"
        entered_pin = self.get_pin()

        if entered_pin == correct_pin:
            self.message_label.setText("PIN poprawny!")
            self.show_image()  # Wyświetl obrazek bez opóźnienia
        else:
            self.message_label.setText("PIN niepoprawny! Spróbuj ponownie.")
            self.clear_pin_fields()  # Wyczyść pola po błędnym PIN-ie
            self.pin_fields[0].setFocus()  # Ustawienie focusu na pierwsze pole

    def show_image(self):
        # Usuwanie poprzednich widgetów
        self.clear_widgets()

        # Wyświetlanie obrazka
        self.label = QLabel(self)
        pixmap = QPixmap('Images/Kostunio.jpg')
        
        if not pixmap.isNull():  # Sprawdzenie, czy obrazek został poprawnie załadowany
            self.label.setPixmap(pixmap)
            self.label.resize(pixmap.width(), pixmap.height())

            # Nowy layout, aby dodać obrazek
            new_layout = QVBoxLayout()
            new_layout.addWidget(self.label)
            self.setLayout(new_layout)

            # Ustawienie rozmiaru okna na rozmiar obrazka
            self.setFixedSize(pixmap.width(), pixmap.height())
            
            # Wycentrowanie okna na ekranie
            self.move(QApplication.desktop().screen().rect().center() - self.rect().center())
        else:
            self.message_label.setText("Nie udało się załadować obrazka.")

    def clear_widgets(self):
        # Usunięcie wszystkich widgetów z layoutu
        while self.layout.count() > 0:
            widget = self.layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()  # Usunięcie widgetu


if __name__ == '__main__':
      
    
    app = QApplication(sys.argv)

    style_sheet = """
    /* Romantyczny styl dla QTextEdit */
    QTextEdit {
        background-color: #FFF0F5; /* Lavender Blush */
        color: #8B0000; /* Dark Red */
        font-family: 'Georgia', serif;
        font-size: 20px;
        border: 2px solid #FFC0CB; /* Pink */
        border-radius: 10px;
        padding: 10px;
    }

    /* Romantyczny styl dla QPushButton */
    QPushButton {
        background-color: #FFB6C1; /* Light Pink */
        color: #FFFFFF; /* White */
        font-family: 'Georgia', serif;
        font-size: 14px;
        border: 2px solid #FF69B4; /* Hot Pink */
        border-radius: 10px;
        padding: 10px 20px;
    }

    QPushButton:hover {
        background-color: #FF69B4; /* Hot Pink */
    }

    QPushButton:pressed {
        background-color: #FF1493; /* Deep Pink */
    }
    """
    app.setStyleSheet(style_sheet)



    window = StartScreen()
    window.setGeometry(750, 250, 340, 300)
    window.show()

    '''
    pyinstaller --onefile --windowed  --add-data "Images;Images" PoemAI.py
    '''

    sys.exit(app.exec_())