import sys

from pyside6x import *


class LabellingWidget(QWidgetExt):
    def __init__(self):
        super().__init__(layout=QHBoxLayout())

        # -- left --

        self.tracks = QColumn(
            QRow(
                QLabel("Directory"),
                QLineEdit(),
                QPushButton("Select"),
            ),
            None,
            QRow(
                QComboBox(),
                None,
                QPushButton("Prev"),
                QPushButton("Next"),
            ),
            QLabel(),
            QRow(
                QPushButton("Prev"),
                QPushButton("Next"),
                None,
                QPushButton("Select"),
            ),
            style=".QWidget { border: 2px solid red; }"
        )

        # -- right --

        self.people = QColumn(
            QRow(
                QLabel("Person"),
                QLineEdit(),
            ),
            QRow(
                QComboBox(),
                None,
                QPushButton("Prev"),
                QPushButton("Next")
            ),
            QLabel(),
            QRow(
                QPushButton("Prev"),
                QPushButton("Next"),
                None,
                QPushButton("Select")
            ),
            style=".QWidget { border: 2px solid blue; }"
        )

        self.addWidgets(self.tracks, self.people)
# end


def main():
    app = QApplication([])

    widget = LabellingWidget()
    widget.resize(1600, 900)
    widget.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

