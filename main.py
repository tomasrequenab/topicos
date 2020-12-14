# !/usr/bin/env python
# coding: utf-8
# name: Tomas Requena
# email: tomas.requena@utp.ac.pa

import sys
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PIL import Image
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from skimage import img_as_float, color
from skimage.color import rgb2gray
from skimage.util import view_as_blocks
import cv2

# Window setup
class App(QMainWindow):
    """
        Se encarga de la configuración general e inicializaciónde la interfaz de usuario (QMainWindow)
    """
    def __init__(self):
        """ Inicializa la instancia de la clase App.
            :param name: self - default param.
            :returns:  QMainWindow -- the return code.
        """
        super().__init__()
        self.title = 'Semestral.py'
        self.left = 0
        self.top = 0
        self.width = 700
        self.height = 200
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.table_widget = WindowContainter(self)
        self.setCentralWidget(self.table_widget)
        self.show()

# Tabs layout
class WindowContainter(QWidget):
    """
        Genera todo el esqueleto de Tabs de la UI, adicionalmente, alberga todos
        los metodos para el procesamiento de imágenes y obtención del archivo a manipular
    """
    def __init__(self, parent):
        """ Inicializa la instancia de la clase WindowContainer y despliega los diferentes tabs en la UI.
            :param name: self - default param.
            :param name: parent - Superclass reference.
            :returns:  QMainWindow -- the return code.
        """
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        self.tabs.resize(300, 200)

        # Add tabs
        self.tabs.addTab(self.tab1, "Inicio")
        self.tabs.addTab(self.tab2, "Colores")
        self.tabs.addTab(self.tab3, "Pooling")
        self.tabs.addTab(self.tab4, "Face Detection")

        # First tab
        self.tab1.layout = QVBoxLayout(self)
        self.tab1.layout.setAlignment(Qt.AlignTop)
        self.welcomeMsg = QLabel('<h1>Escoge la imagen que deseas procesar:</h1>')
        self.input_layout = QHBoxLayout()
        self.input_text = QLineEdit()
        self.get_file_btn = QPushButton('Examinar')
        self.get_file_btn.clicked.connect(self.get_file_image)
        self.input_layout.addWidget(self.input_text)
        self.input_layout.addWidget(self.get_file_btn)
        show_file_btn = QPushButton('Ver Imagen')
        show_file_btn.clicked.connect(self.show_images)
        self.tab1.layout.addWidget(self.welcomeMsg)
        self.tab1.layout.addLayout(self.input_layout)
        self.tab1.layout.addWidget(show_file_btn)
        self.tab1.setLayout(self.tab1.layout)

        # Second tab
        self.tab2.layout = QVBoxLayout(self)
        self.tab2.layout.setAlignment(Qt.AlignTop)
        self.tab2.layout.addWidget(QLabel('<h3>Filtros de color en la imagen</h3>'))
        self.primary_colors = QHBoxLayout()
        self.secundary_colors = QHBoxLayout()
        self.tab2.layout.addLayout(self.primary_colors)
        self.tab2.layout.addLayout(self.secundary_colors)
        self.tab2.setLayout(self.tab2.layout)

        # Third tab
        self.tab3.layout = QVBoxLayout(self)
        self.tab3.layout.setAlignment(Qt.AlignTop)
        self.tab3.layout.addWidget(QLabel('<h3>Filtros de Pooling en la imagen</h3>'))
        self.first_pooling = QHBoxLayout()
        self.second_pooling = QHBoxLayout()
        self.tab3.layout.addLayout(self.first_pooling)
        self.tab3.layout.addLayout(self.second_pooling)
        self.tab3.setLayout(self.tab3.layout)

        # Fourth tab
        self.tab4.layout = QVBoxLayout(self)
        self.tab4.layout.setAlignment(Qt.AlignTop)
        self.tab4.layout.addWidget(QLabel('<h3>Deteccion de Rostro en la imagen</h3>'))
        self.tab4.setLayout(self.tab4.layout)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)

    # Getting image filepath
    def get_file_image(self):
        """ Obtiene la ruta del archivo seleccionado en el explorador de directorios.
            :param name: self - default param.
            :returns: no explicit returns - assign self.input_text (str).
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Escoge una imagen', '', 'JPEG File (*.jpeg);;JPG File (*.jpg);;PNG File (*.png)', options=options)
        if file_path:
            self.input_text.setText(file_path)

    # Show preview image and calling proc methods
    def show_images(self):
        """ Muestra la imagen original seleccionada en la UI y ejecuta los metodos de procesamiento.
            :param name: self - default param.
            :returns: no explicit returns - generate pixmap for image (QLabel).
        """
        image_path = self.input_text.text()
        self.image = QLabel(self)
        pixmap = QPixmap(image_path)
        resized = pixmap.scaled(int(pixmap.width() / 1.5), int(pixmap.height() / 1.5))
        self.image.setPixmap(resized)
        self.image.setAlignment(Qt.AlignCenter)
        self.tab1.layout.addWidget(self.image)

        self.show_colors_image()
        self.show_poolings_image()
        self.detect_face()

    # Processing RGB images
    def show_colors_image(self):
        """ Genera 6 imagenes con los respectivos colores primarios y secundarios en el Tabs Colores.
            :param name: self - default param.
            :returns: no explicit returns - generate a figure of six images (color_figure).
        """
        image_array = self.get_array_image()
        float_image = img_as_float(image_array[::2, ::2])
        image = color.gray2rgb(float_image)

        # RBG config
        color_images = [
            ("Amarillo", [1, 1, 0]),
            ("Azul", [0, 0, 1]),
            ("Rojo", [1, 0, 0]),
            ("Verde", [0, 1, 0]),
            ("Morado", [1, 0, 1]),
            ("Naranja", [1, 0.5, 0]),
        ]

        # Drawing images
        color_figure, color_axes = plt.subplots(2, 3, sharex=True, sharey=True)
        ax = color_axes.ravel()

        for i, a in enumerate(ax):
            width, height = self.get_ax_size(a, color_figure)
            a.set_title(color_images[i][0])
            a.imshow((image * color_images[i][1]), extent=(0, width, height, 0), cmap=cm.Greys_r) #
            a.set_axis_off()

        color_figure.tight_layout()
        color_canvas = FigureCanvasQTAgg(color_figure)
        self.tab2.layout.addWidget(color_canvas)

    # Processing pooling images
    def show_poolings_image(self):
        """ Genera 4 imagenes con diferentes grados de agrupación (pooling) en el Tabs Pooling.
            :param name: self - default param.
            :returns: no explicit returns - generate a figure of four images (pooling_figure).
        """
        image_array = self.get_array_image()
        gray_image = rgb2gray(image_array)
        block_shape = (4, 4)
        block_view = view_as_blocks(gray_image, block_shape)
        flatten_view = block_view.reshape(block_view.shape[0], block_view.shape[1], -1)

        # Proccesing images
        mean_view = np.mean(flatten_view, axis=2)
        max_view = np.max(flatten_view, axis=2)
        median_view = np.median(flatten_view, axis=2)
        gray_resized = ndi.zoom(gray_image, 2, order=3)

        # Drawing images
        pooling_figure, pooling_axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
        ax = pooling_axes.ravel()

        pooling_images = [
            ("Original", gray_resized),
            ("Mean pooling", mean_view),
            ("Max pooling", max_view),
            ("Median pooling", median_view)
        ]

        for i, a in enumerate(ax):
            width, height = self.get_ax_size(a, pooling_figure)
            a.set_title(pooling_images[i][0])
            a.imshow(pooling_images[i][1], extent=(0, width, height, 0),  cmap=cm.Greys_r)
            a.set_axis_off()

        pooling_figure.tight_layout()
        pooling_canvas = FigureCanvasQTAgg(pooling_figure)
        self.tab3.layout.addWidget(pooling_canvas)

    # Detecting face in image
    def detect_face(self):
        """ Genera y guarda una imagen con la cara identificada en el Tabs Face Dectection.
            :param name: self - default param.
            :returns: no explicit returns - display and generate an image (face_figure).
        """
        face_cascade = cv2.CascadeClassifier('face_detection.xml')
        face_img = cv2.imread(self.input_text.text())
        faces = face_cascade.detectMultiScale(face_img, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(face_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite("face_detected.png", face_img)

        # Drawing image
        face_figure = Figure()
        face_axes = face_figure.add_subplot(111)
        face_axes.axis('off')
        face_axes.imshow(face_img)
        face_canvas = FigureCanvasQTAgg(face_figure)
        self.tab4.layout.addWidget(face_canvas)

    # Processing image to np.array
    def get_array_image(self):
        """ Convierte la imagen en una matrix Numpy
            :param name: self - default param.
            :returns: image_array - np.array.
        """
        file_path = self.input_text.text()
        img = Image.open(file_path)
        image_array = np.asarray(img)
        return image_array

    # Getting axes size (w, h)
    def get_ax_size(self, ax, fig):
        """ Calcula el tamaño de un ax de una figura dada
            :param name: self - default param.
            :param name: ax - ax (Matplotlib).
            :param name: fig - Figure (Matplotlib).
            :returns: width, height - tuple.
        """
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        width *= fig.dpi
        height *= fig.dpi
        return width, height

    @pyqtSlot()
    def on_click(self):
        """ Permite la navegación entre tabs
            :param name: self - default param.
        """
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())

# Main
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

