import os
import numpy as np

class Buscador():

    def __init__(self, path, numero_clases):
        self.path = path
        self.numero_clases = numero_clases
        self.X_data = []
        self.y_labels = []

    def encuentra_conjuntos_entrenamiento_test_desde_path(self):
        for root, dirs, files in os.walk(self.path):
            for file_name in files:
                if (file_name.endswith(".png")):
                    full_path = os.path.join(root, file_name)
                    print(full_path)
                    self._obtiene_conjuntos_desde_path_de_señales(full_path)

    def _obtiene_conjuntos_desde_path_de_señales(self, path):
        
        #If path contains 'train', y_label is two dir up. Else if path contains 'test', y_label is one dir up.
        #:param path: the full path

        y_label_hot_true = np.zeros(self.numero_clases, dtype=np.float32)
                
        if '0' in path:
           y_dir = os.path.dirname(path)
           y_label = os.path.basename(y_dir)
           y_label_hot_true[int(y_label)] = 1
           self.X_data.append(path)
           self.y_labels.append(list(y_label_hot_true))
           print(y_label_hot_true)
           print()
        elif '1' in path:
           y_dir = os.path.dirname(path)
           y_label = os.path.basename(y_dir)
           y_label_hot_true[int(y_label)] = 1
           self.X_data.append(path)
           self.y_labels.append(list(y_label_hot_true))
           print(y_label_hot_true)
           print()
        else:
           print("no")

