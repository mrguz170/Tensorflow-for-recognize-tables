import Modelo_Convolucional_No_GPU as modelo
import busc
from TensorFlowUtils import pt
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image


PATH_TO_TEST_IMAGES_DIR = "img\\Pruebas"
image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, '2.png') 
#etiqueta correcta de la prueba
y_feed = [[1, 0]]

def load_test_images():
    '''
    :return: Tuple of (test images, image labels)
    '''

    """
    Ubicaciones de las imagenes para entrenar el modelo
    """
    conjunto_entrenamiento =    "C:\\Users\\lobo_\\Documents\\Machine Learning\\TDI\\img\\classification sets\\train\\"
    conjunto_test =             "C:\\Users\\lobo_\\Documents\\Machine Learning\\TDI\\img\\classification sets\\test\\"

    numero_clases = 2 # Start in 0

    searcher = busc.Buscador(path=[conjunto_entrenamiento, conjunto_test], numero_clases=numero_clases)
    searcher.encuentra_conjuntos_entrenamiento_test_desde_path()

    train_set = [searcher.x_train, searcher.y_train]  # Train Set
    test_set = [searcher.x_test, searcher.y_test]  # Test Set

    return searcher.x_test, searcher.y_test
   


def load_and_predict_with_checkpoints(image_np_expanded):
    '''
    Loads saved model checkpoints and make prediction on test images
    '''
    
    save_dir = './models/model_TDI/'
    version = '02'
    dir = os.path.join(save_dir, version)

    # create an empty graph for the session
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # restore save model
        saver = tf.train.import_meta_graph( dir + '/model{}.ckpt.meta'.format(version))
        saver.restore(sess, tf.train.latest_checkpoint(dir))

        # get necessary tensors by name
        pred_class_tensor = loaded_graph.get_tensor_by_name("y_prediction:0")
        inputs_real_tensor = loaded_graph.get_tensor_by_name("x_real:0")
        y_tensor = loaded_graph.get_tensor_by_name("y_:0")
        drop_rate_tensor = loaded_graph.get_tensor_by_name("keep_probably:0")
        correct_pred_sum_tensor = loaded_graph.get_tensor_by_name("correct_prediction:0")


        # make prediction
        correct, pred_class = sess.run(
            [correct_pred_sum_tensor, pred_class_tensor],
            feed_dict={
                inputs_real_tensor: image_np_expanded,
                y_tensor: y_feed,
                drop_rate_tensor:0.5 })

        # print results
        print("No. correct predictions: {}".format(correct))
        print("Predicted classes: {}".format(pred_class))



"""
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
"""



def load_image_into_numpy_array(image):
  
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 1)).astype(np.uint8)




"""
MAIN
"""
def main(_):

    height=60
    width=60

    img = cv2.imread(image_path, 0)
    #img = cv2.resize(img, (height, width))
    img = img.reshape(1, 3600)
    #img = Image.open(image_path)
    
    #image_np_expanded = load_image_into_numpy_array(img)

    #x = tf.placeholder(tf.float32, shape=[None, self.input_rows_numbers * self.input_columns_numbers], name='x_real') 

    #image_np_expanded = tf.reshape(img, [-1, 60, 60, 1])
    

   # image_np_expanded = np.expand_dims(image_np, axis=0)

    load_and_predict_with_checkpoints(img)
 

if __name__ == '__main__':
    tf.app.run()
