import numpy
import cv2
import PIL
import matplotlib.pyplot as plt
from PIL import Image
from keras.applications import VGG16
from keras.callbacks import EarlyStopping
from keras.constraints import maxnorm
from keras.engine import Layer, merge, Merge, Input, Model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from matplotlib import pyplot
from pip.req.req_file import process_line
from scipy.misc import toimage
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Reshape, Permute, Activation, Dropout
from keras.optimizers import RMSprop, SGD
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, UpSampling2D, Deconvolution2D, Cropping2D
from keras.utils import np_utils
from keras import backend as k

k.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)
train_directory = 'data/train'
test_directory = 'data/test'
filepath = "data/models"
train_datagen = ImageDataGenerator(

            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
train_data_generator = train_datagen.flow_from_directory(

            train_directory,
            target_size=(150, 150),
            batch_size=96,
            class_mode=None
      )


def train_generator():
    for a in train_data_generator:
        yield a, a


test_datagen = ImageDataGenerator(

            rescale=1./255
        )

test_data_generator = test_datagen.flow_from_directory(

    test_directory,
    target_size=(150, 150),
    batch_size=96,
    class_mode=None
)


def test_generator():
    for b in test_data_generator:
        yield b, b

input_img = Input(shape=(3, 150, 150))
cnn1 = Convolution2D(96, 3, 3, activation='relu', border_mode='same', name='input')(input_img)
cnn2 = Convolution2D(96, 3, 3, activation='relu', border_mode='same', name="conv_net1")(cnn1)
pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(cnn2)

cnn3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv_net2_1')(pool1)
cnn4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv_net2_2')(cnn3)
pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(cnn4)

cnn5 = Convolution2D(384, 3, 3, activation='relu', border_mode='same', name='conv_net3_1')(pool2)
cnn6 = Convolution2D(384, 3, 3, activation='relu', border_mode='same', name='conv_net3_2')(cnn5)
cnn7 = Convolution2D(384, 3, 3, activation='relu', border_mode='same', name='conv_net3_3')(cnn6)
pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(cnn7)

cnn8 = Convolution2D(384, 3, 3, activation='relu', border_mode='same', name='conv_net4_1')(pool3)
cnn9 = Convolution2D(384, 3, 3, activation='relu', border_mode='same', name='conv_net4_2')(cnn8)
cnn10 = Convolution2D(384, 3, 3, activation='relu', border_mode='same', name='conv_net4_3')(cnn9)
pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4')(cnn10)

cnn11 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv_net5_1')(pool4)
cnn12 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv_net5_2')(cnn11)
cnn13 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv_net5_3')(cnn12)
pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5')(cnn13)

cnn14 = Convolution2D(4096, 1, 1, activation='relu', border_mode='same', name='conv_net6_1')(pool5)
dropout1 = Dropout(0.5)(cnn14)
cnn15 = Convolution2D(4096, 1, 1, activation='relu', border_mode='same', name='conv_net7_2')(dropout1)
dropout2 = Dropout(0.5)(cnn15)
cnn16 = Convolution2D(21, 1, 1, activation='relu', border_mode='same', name='conv_net8_3')(dropout2)

decnn1 = Deconvolution2D(21, 1, 1, output_shape=(None, 21, 8, 8), border_mode='valid', subsample=(2, 2))(cnn16)
crop1 = Cropping2D(cropping=((2, 2), (2, 2)))(decnn1)

X = Convolution2D(21, 1, 1, activation='relu', border_mode='same', name='conv_net8_1')(pool5)
crop2 = Cropping2D(cropping=((2, 2), (2, 2)))(X)
merging = merge([crop1, X], mode='sum')

decnn2 = Deconvolution2D(21, 16, 16, output_shape=(None, 21, 158, 158), border_mode='valid', subsample=(8, 8))(merging)
crop3 = Cropping2D(cropping=((4, 4), (4, 4)))(decnn2)
reshape = Reshape((21, 150*150))(crop3)
activation = Activation('softmax')(reshape)
model = Model(input_img, activation)

epochs = 100
learning_rate = 0.001
decay = learning_rate / epochs
sgd = SGD(lr=learning_rate, decay=decay, momentum=0.9, nesterov=True)
rms = RMSprop(lr=learning_rate, decay=decay)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
print(model.summary())
history = model.fit_generator(
    train_generator(),
    samples_per_epoch=256,
    nb_epoch=100,
    validation_data=test_generator(),
    nb_val_samples=4
)

#model.load_weights(filepath=filepath+"/objectdetector.h5")
file = open(filepath+"/objectdetector.h5", 'a')
model.save_weights(filepath=filepath+"/objectdetector.h5")
file.close()

img = Image.open("data/test/shadows/IMG_2157.jpg")
img = img.resize((150, 150), PIL.Image.ANTIALIAS)
img.save("data/test/shadows/IMG_2157.jpg")
# i = cv2.imread("data/train/shadows/test/nonshadows/04822305_257_1025_513_1281.jpg")
# cv2.putText(i, "Shadow Detected", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# cv2.imshow("Object", i)
# cv2.waitKey(100000)
testImage = load_img("data/test/shadows/IMG_2157.jpg")
x = img_to_array(testImage)
x = x.reshape((1,)+x.shape)
model.predict(x)
