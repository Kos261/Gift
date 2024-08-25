from PyQt5.QtWidgets import QApplication, QGridLayout, QLineEdit, QWidget
from PyQt5.QtCore import Qt

class PinWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 100)
        
        # Utworzenie siatki layoutu
        self.layout = QGridLayout(self)
        
        # Lista do przechowywania referencji do pól QLineEdit
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

    def focus_next_field(self, index):
        if index < 7:  # Przechodzimy do kolejnego pola tylko jeśli nie jesteśmy w ostatnim
            self.pin_fields[index + 1].setFocus()

    def get_pin(self):
        # Pobranie PIN-u z pól
        pin = ''.join([field.text() for field in self.pin_fields])
        return pin

if __name__ == "__main__":
    app = QApplication([])
    widget = PinWidget()
    widget.show()
    app.exec_()
