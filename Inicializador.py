"""
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
"""
import Modelo_Convolucional_No_GPU as modelo
import busc
from TensorFlowUtils import pt
import tensorflow as tf

"""
Ubicaciones de las imagenes para entrenar el modelo
"""

conjunto_entrenamiento =    "C:\\Users\\lobo_\\Documents\\Machine Learning\\TDI\\img\\classification sets\\sets\\train\\"
conjunto_test =             "C:\\Users\\lobo_\\Documents\\Machine Learning\\TDI\\img\\classification sets\\sets\\test\\"

numero_clases = 2 # Start in 0

searcher = busc.Buscador(path=[conjunto_entrenamiento, conjunto_test], numero_clases=numero_clases)
searcher.encuentra_conjuntos_entrenamiento_test_desde_path()

percentages_sets = None  # Example

"""
Getting train, validation (if necessary) and test set.
"""
train_set = [searcher.x_train, searcher.y_train]  # Train Set
test_set = [searcher.x_test, searcher.y_test]  # Test Set


pt(test_set[0])
pt(test_set[1])


del searcher

option_problem = "Problema Mesas"

models = modelo.Modelo(input=train_set[0],test=test_set[0],
                         input_labels=train_set[1],test_labels=test_set[1],
                         number_of_classes=numero_clases,
                         option_problem=option_problem)

models.convolucion_imagenes()



