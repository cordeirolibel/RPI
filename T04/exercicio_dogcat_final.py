
# coding: utf-8

# In[ ]:

import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm
from random import shuffle
import os
from sklearn.preprocessing import StandardScaler  

#---------------------------------------------
#Settings: definição dos parêmetros da rede:
n_classes = 2                     # A base de dados DogsAndCats tem 2 classes de objetos!!!!
nepochs = 10                      # Número de épocas para o treinamento!!!
batch_size = 32                   # Número de imagens por batch!!!
image_size = 32                   # Todas as imagens devem ser redimensionadas para 32x32 pixels!!!
nchannels = 3                     #
n_input = image_size * image_size * nchannels # Tamanho da entrada!
learning_rate = 1e-3              # Taxa de aprendizado!!!
TRAIN_DIR = './dog-cat/train'
TEST_DIR = './dog-cat/test'

#---------------------------------------------
def multilayer_perceptron (x):
    n_camada_1 = 256 # Número de neurônios na camada 1!!!
    n_camada_2 = 128 # Número de neurônios na camada 2!!!
   
    # Primeira camada da rede:
    W1 = tf.get_variable('w1', [n_input, n_camada_1], initializer = tf.random_normal_initializer())
    b1 = tf.get_variable('b1', [n_camada_1], initializer = tf.random_normal_initializer())
    y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1) #tf.nn.relu() or tf.matmul(x, W1) + b1!!!! 

    # Segunda camada da rede:
    W2 = tf.get_variable('w2', [n_camada_1, n_camada_2], initializer = tf.random_normal_initializer())
    b2 = tf.get_variable('b2', [n_camada_2], initializer = tf.random_normal_initializer())
    y2 = tf.nn.sigmoid(tf.matmul(y1, W2) + b2) 

    # Última camada da rede:
    W3 = tf.get_variable('w3',[n_camada_2, n_classes], initializer = tf.random_normal_initializer())
    b3 = tf.get_variable('b3',[n_classes], initializer = tf.random_normal_initializer())
    out_layer = tf.matmul(y2, W3) + b3 

    return out_layer

#---------------------------------------------
def create_label(image_name):
    word_label = image_name.split('.')[-3]
    if word_label == 'cat':
        return np.array([1,0])
    elif word_label == 'dog':
        return np.array([0,1])
    else: 
        print ("Esta classe não existe!!!!!")

#---------------------------------------------
def read_dataset (filename):
    X = []
    Y = []
    for img in tqdm(os.listdir(filename)):
        path = os.path.join(filename, img)
        img_data = cv2.imread(path)
        #img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #converte para níveis de cinza
        img_data = cv2.resize(img_data, (image_size, image_size)) #deixa com as dimensões definidas
        
        #primeira forma de normalização
        #min_val = np.min(img_data)
        #max_val = np.max(img_data )
        #img_data = (img_data-min_val)/(max_val-min_val)
        #segunda forma de normalização
        img_data = img_data/255.0    
        
        #cria os vetores de dados e de labels
        X.append(np.array(img_data))
        Y.append(np.array(create_label(img)))
        
    return X,Y

#---------------------------------------------
def next_batch (num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    labels_shuffle = labels[idx]
    return data_shuffle, labels_shuffle

#---------------------------------------------
if __name__ == "__main__":

    # Leitura da base de dados:
    X_train,Y_train = read_dataset (TRAIN_DIR)
    X_train = np.asarray(X_train).reshape(-1, n_input)
    Y_train = np.asarray(Y_train)

    X_test,Y_test = read_dataset (TEST_DIR)
    X_test = np.asarray(X_test).reshape(-1, n_input)
    Y_test = np.asarray(Y_test)

    #terceira forma de normalização
    #ATENÇÃO: para testar, comente todas as formas de normalização da função read_dataset()
    #scaler = StandardScaler()  
    #scaler.fit(X_train)  
    #X_train = scaler.transform(X_train)  
    #X_test = scaler.transform(X_test)  

    # Variáveis do tensorflow:
    Y = tf.placeholder(tf.float32, [None, n_classes])
    X = tf.placeholder(tf.float32, [None, image_size * image_size * nchannels])
   
    Ypred = multilayer_perceptron (X) 
  
    # Funções de custo:
    error1 = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(tf.nn.softmax(Ypred)), reduction_indices = [1]))
    error2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Ypred, labels = Y)) 
    error3 = tf.reduce_mean(tf.reduce_sum(tf.square(tf.nn.softmax(Ypred) - Y), reduction_indices = [1]))
    error = error2
    
    # Funções para minimização de erro: 
    optimizer1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(error)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)
    optimizer = optimizer2
 
    corr = tf.equal(tf.argmax(Ypred,1),tf.argmax(Y,1))
 
    accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))

    # Inicialização de variáveis:
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        # Treino:
        for epoch in range(nepochs):
            train_err = 0
            train_acc = 0
            train_batches = 0
            total_batch = int(len(X_train)/batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = next_batch (batch_size, X_train, Y_train)
                sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
                err, acc = sess.run([error,accuracy], feed_dict={X: batch_xs, Y: batch_ys})
                train_err += err
                train_acc += acc
                train_batches += 1
            print("Epoch: ", '%2d' % (epoch+1))
            print("  training loss:\t\t{:.6f}".format(train_err/train_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(train_acc/train_batches * 100))

        # Testes:
        test_err = 0
        test_acc = 0
        test_batches = 0
        total_batch = int(len(X_test)/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch (batch_size, X_test, Y_test)
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
            err, acc = sess.run([error,accuracy], feed_dict={X: batch_xs, Y: batch_ys})
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err/test_batches))
        print("  test accuracy:\t\t{:.2f} %".format((test_acc/test_batches)*100))

