import random
import sys

import PySide6.QtCore
from PySide6 import QtCore

# Prints PySide6 version
print(PySide6.__version__)

# Prints the Qt version used to compile PySide6
print(PySide6.QtCore.__version__)

from pyside6x import *


class MyWidget(QWidgetExt):
    def __init__(self):
        super().__init__(
            title="MyWidget",
            layout=QtWidgets.QVBoxLayout()
        )

        self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]

        self.button = QtWidgets.QPushButton("Click me!")
        self.button2 = QtWidgets.QPushButton("Click you!")

        self.text = QtWidgets.QLabel("Hello World", alignment=QtCore.Qt.AlignCenter)

        self.addWidgets(
            self.text,
            QRow(self.button, self.button2)
        )

        self.button.clicked.connect(self.magic)
        self.button2.clicked.connect(self.magic2)

    @QtCore.Slot()
    def magic(self):
        self.text.setText(random.choice(self.hello))

    @QtCore.Slot()
    def magic2(self):
        self.text.setText(random.choice(self.hello))


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(1600, 900)
    widget.show()

    sys.exit(app.exec())
