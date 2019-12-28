from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Input, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras import backend as K
from keras.applications import VGG16, imagenet_utils
from keras.optimizers import Adam
from keras.models import Model, load_model

#---------------------------------------------
#Settings: definicao dos parametros da rede:
n_classes = 10                     # A base de dados Cifar10 tem 10 classes de objetos!!!!
nepochs = 30                       # Numero de epocas para o treinamento!!!
batch_size = 128                   # Numero de imagens por batch!!!
image_size = 32                    # Todas as imagens devem ser redimensionadas para 32x32 pixels!!!
nchannels = 3                      # Numero de canais na imagem!!!
learning_rate = 1e-4               # Taxa de aprendizado!!!
keep_probability = 0.5             # Taxa de dropout!!!
TRAIN_DIR = './cifar10-keras/treino/'
TEST_DIR = './cifar10-keras/teste/'
cnn_last_layer_length = 256

#---------------------------------------------
def get_cnn_model ():

    input_shape = (image_size, image_size, nchannels)

    model = Sequential()

    model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    
    model.add(Dense(cnn_last_layer_length, activation='relu', name='fc1'))
    model.add(Dropout(keep_probability))
    model.add(Dense(cnn_last_layer_length, activation='relu', name='fc2'))
    model.add(Dropout(keep_probability))
    
    model.add(Dense(n_classes, activation='softmax'))
    
    return model

#---------------------------------------------
def get_vgg_model ():

    input_shape = (image_size, image_size, nchannels)
    input_tensor = Input (shape=input_shape)
    baseModel = VGG16 (weights='imagenet', include_top=False, input_tensor=input_tensor)

    modelStruct = baseModel.output
    modelStruct = Flatten(input_shape=baseModel.output_shape[1:])(modelStruct)
    
    modelStruct = Dense(cnn_last_layer_length, activation='relu', name='fc1')(modelStruct)
    modelStruct = Dropout(keep_probability)(modelStruct)
    modelStruct = Dense(cnn_last_layer_length, activation='relu', name='fc2')(modelStruct)
    modelStruct = Dropout(keep_probability )(modelStruct)
    predictions = Dense(n_classes, activation='softmax')(modelStruct)

    model = Model(input=[baseModel.input], output=predictions)

    for i,layer in enumerate(model.layers):
        layer.trainable = True

    return model

if K.image_data_format() == 'channels_first':
    input_shape = (3, image_size, image_size)
else:
    input_shape = (image_size, image_size, 3)

#model = get_cnn_model ()
model = get_vgg_model ()

model.compile(optimizer=Adam(lr=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
                  rescale = 1./255, 
                  horizontal_flip = True, 
                  vertical_flip = True, 
                  width_shift_range=0.2,
                  height_shift_range=0.2,
                  zoom_range = [0.9, 1.0]
               )

test_datagen = ImageDataGenerator (
                  rescale=1./255
               )

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical')

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

test_generator = test_datagen.flow_from_directory (
    TEST_DIR,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical')

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

model.fit_generator (
    train_generator,
    steps_per_epoch = STEP_SIZE_TRAIN,
    epochs=nepochs,
    validation_data = test_generator,
    validation_steps = STEP_SIZE_TEST
)

model.save('model.h5')

