import sys
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from solution import sol_model


class MyWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('win.ui', self)
        self.pushButton.clicked.connect(self.load_img)
        self.pushButton_2.clicked.connect(self.sol)

    def load_img(self):
        file_name = QFileDialog.getOpenFileName()[0]
        self.path_img = file_name
        pixmap = QPixmap(file_name)
        self.label.setPixmap(pixmap.scaled(391, 231))

    def sol(self):
        sol_model(self.path_img)

        path_image = 'image_detect.png'
        pixmap = QPixmap(path_image)
        self.label_2.setPixmap(pixmap.scaled(400, 64))

        path_label = 'label_detect.png'
        pixmap = QPixmap(path_label)
        self.label_3.setPixmap(pixmap.scaled(400, 64))


app = QApplication(sys.argv)
ex = MyWidget()
ex.setWindowTitle('Детекция металла')
ex.show()
sys.exit(app.exec_())
