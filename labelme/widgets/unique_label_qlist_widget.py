# -*- encoding: utf-8 -*-

import html

from qtpy import QtWidgets
from qtpy.QtCore import Qt

from .escapable_qlist_widget import EscapableQListWidget


class CustomListWidgetItem(QtWidgets.QListWidgetItem):
    def __init__(self, sort_key):
        super().__init__()
        self.sort_key = sort_key

    def __lt__(self, other):
        return self.sort_key < other.sort_key


class UniqueLabelQListWidget(EscapableQListWidget):
    def mousePressEvent(self, event):
        super(UniqueLabelQListWidget, self).mousePressEvent(event)
        if not self.indexAt(event.pos()).isValid():
            self.clearSelection()

    def findItemByLabel(self, label):
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.UserRole) == label:
                return item

    def createItemFromLabel(self, label):
        if self.findItemByLabel(label):
            raise ValueError(
                "Item for label '{}' already exists".format(label)
            )

        item = CustomListWidgetItem(label)
        item.setData(Qt.UserRole, label)
        return item

    def setItemLabel(self, item, label, color=None):
        qlabel = QtWidgets.QLabel()
        if color is None:
            qlabel.setText("{}".format(label))
        else:
            qlabel.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    html.escape(label), *color
                )
            )
        qlabel.setAlignment(Qt.AlignBottom)

        item.setSizeHint(qlabel.sizeHint())

        self.setItemWidget(item, qlabel)

    def prev_label(self):
        print("prev_label")
        current_row = self.currentRow()
        if current_row > 0:
            self.setCurrentRow(current_row - 1)

    def next_label(self):
        print("next_label")
        current_row = self.currentRow()
        if current_row < self.count() - 1:
            self.setCurrentRow(current_row + 1)

        # key = ev.key()
        # key_name = QtGui.QKeySequence(key).toString()
        # try:
        #     print(f'keyPressEvent {key_name} canvas self.drawing()', self.drawing())
        # except:
        #     pass
        # if key == QtCore.Qt.Key_Up:
        #     current_row = self.currentRow()
        #     if current_row > 0:
        #         self.setCurrentRow(current_row - 1)
        # elif key == QtCore.Qt.Key_Down:
        #     current_row = self.currentRow()
        #     if current_row < self.count() - 1:
        #         self.setCurrentRow(current_row + 1)
    #
    # def keyPressEvent(self, ev):
    #     self.check_key_press(ev)
    #     super(UniqueLabelQListWidget, self).keyPressEvent(ev)
