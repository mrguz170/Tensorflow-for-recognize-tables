import busc2
from TensorFlowUtils import pt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import cv2

data =    "C:\\Users\\lobo_\\Documents\\Machine Learning\\TDI\\img\\classification sets\\data\\"
numero_clases = 2

print("Iniciando!")
searcher = busc2.Buscador(path=data, numero_clases=numero_clases)
searcher.encuentra_conjuntos_entrenamiento_test_desde_path()
print("Imagenes cargadas!")

x_data, y_label = searcher.X_data, searcher.y_labels  # Train Set

X_train, X_test, y_train, y_test = train_test_split(x_data,y_label,test_size=0.33, random_state=101)

for x in X_test:
    image = cv2.imread(x, 1)  
    path = "classification sets\\sets\\test\\"
    cv2.imwrite(os.path.join(path, "{}".format(x)), image)