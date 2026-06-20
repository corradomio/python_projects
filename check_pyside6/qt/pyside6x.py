#
#
from PySide6.QtWidgets import *


class QSpacer(QSpacerItem):
    def __init__(self, layout: QLayout):
        super().__init__(
            1,1,
            hData = (QSizePolicy.Policy.Expanding
                     if isinstance(layout, QHBoxLayout)
                     else QSizePolicy.Policy.Minimum),
            vData = (QSizePolicy.Policy.Expanding
                     if isinstance(layout, QVBoxLayout)
                     else QSizePolicy.Policy.Minimum),
        )

class QHSpacer(QSpacerItem):
    def __init__(self):
        super().__init__(
            1,1,
            hData=QSizePolicy.Policy.Expanding,
            vData=QSizePolicy.Policy.Minimum
        )

class QHVSpacer(QSpacerItem):
    def __init__(self):
        super().__init__(
            1,1,
            hData=QSizePolicy.Policy.Minimum,
            vData=QSizePolicy.Policy.Expanding
        )


# ---------------------------------------------------------------------------

class QWidgetExt(QWidget):
    def __init__(
        self,
        parent:QWidget | None=None,
        *,
        title:str|None=None,
        layout: QLayout|None=None,
        children: list[QWidget]|tuple[QWidget,...]|None=None,
        style: str|None=None,
        frameShape=None
    ):
        super().__init__(parent)

        self.layout = layout

        if title is not None:
            self.setWindowTitle(title)

        if layout is not None:
            self.setLayout(layout)

        if style is not None:
            self.setStyleSheet(style)

        if children is not None:
            self.addWidgets(*children)

    # end

    def addWidgets(self, *widgets):
        if self.layout is None:
            return

        for widget in widgets:
            if isinstance(widget, str):
                self.layout.addItem(QSpacer(self.layout))
            elif widget is None:
                self.layout.addItem(QSpacer(self.layout))
            elif isinstance(widget, QWidgetItem):
                self.layout.addItem(widget)
            elif isinstance(widget, QWidget):
                self.layout.addWidget(widget)
            else:
                raise ValueError(f" Unsupported Widget {widget}")
    # end


class QColumnWidget(QWidgetExt):
    def __init__(self,
                 # parent:QWidget | None=None,
                 # title:str|None=None,
                 *childs, style=None):
        super().__init__(
            parent=None,
            title=None,
            layout=QVBoxLayout(),
            children=childs,
            style=style
        )

QColumn = QColumnWidget


class QRowWidget(QWidgetExt):
    def __init__(self,
                 # parent:QWidget | None=None,
                 # title:str|None=None,
                 *childs, style=None):
        super().__init__(
            parent=None,
            title=None,
            layout=QHBoxLayout(),
            children=childs,
            style=style
        )

QRow = QRowWidget

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
