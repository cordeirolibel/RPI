import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm
from random import shuffle
import os

#---------------------------------------------
#Settings: definição dos parêmetros da rede:
n_classes = ??                    # A base de dados Cifar10 tem 10 classes de objetos!!!!
nepochs = ??                      # Número de épocas para o treinamento!!!
batch_size = ??                   # Número de imagens por batch!!!
image_size = 32                   # Todas as imagens devem ser redimensionadas para 32x32 pixels!!!
nchannels = 3                     # Canais de cores!!!
n_input = image_size * image_size * nchannels # A base de dados Cifar10 tem imagens com dimensões: 32*32 pixels!!!
learning_rate = 1e-3              # Taxa de aprendizado!!!
TRAIN_DIR = './cifar10/cifar_1/'
TEST_DIR = './cifar10/cifar_2/'

#---------------------------------------------
def create_label(image_name):
   word_label = image_name.split('.')[-4]
   if word_label == 'airplane':
      #TO-DO: adicione as classes!!!!  

#---------------------------------------------
def read_dataset (filename):
   X = []
   Y = []
   for img in tqdm(os.listdir(filename)):
      path = os.path.join(filename, img)
      img_data = cv2.imread(path)
      img_data = cv2.resize(img_data, (image_size, image_size))
      #min_val = np.min(img_data)
      #max_val = np.max(img_data )
      #img_data = (img_data-min_val)/(max_val-min_val)
      img_data = img_data/255.0
      X.append(np.array(img_data))
      Y.append(np.array(create_label(img)))
   return X,Y

#---------------------------------------------
if __name__ == "__main__":

   # Leitura da base de dados:
   X_train,Y_train = read_dataset (TRAIN_DIR)
   X_train = np.asarray(X_train).reshape(-1, n_input)
   Y_train = np.asarray(Y_train)
   
   X_test,Y_test = read_dataset (TEST_DIR)
   X_test = np.asarray(X_test).reshape(-1, n_input)
   Y_test = np.asarray(Y_test)

   # TO-DO: adicione codigo para treinar e testar a rede perceptron!!!!
