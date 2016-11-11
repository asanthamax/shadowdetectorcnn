import numpy
import cv2
import PIL
import matplotlib.pyplot as plt
from PIL import Image
from keras.callbacks import EarlyStopping
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from matplotlib import pyplot
from scipy.misc import toimage
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop, SGD
from keras.layers.convolutional import Convolution2D
from keras.utils import np_utils
from keras import backend as k

from datapreprocessor import ImagePreprocessor

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
            batch_size=32,
            class_mode='binary'
      )
test_datagen = ImageDataGenerator(

            rescale=1./255
        )

test_data_generator = test_datagen.flow_from_directory(

    test_directory,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 150, 150), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(2, 2), border_mode='same', W_constraint=maxnorm(3)))
model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(2, 2), border_mode='same', W_constraint=maxnorm(3)))
model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(2, 2), border_mode='same', W_constraint=maxnorm(3)))
model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(2, 2), border_mode='same', W_constraint=maxnorm(3)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(Flatten())
model.add(Dense(64, activation='relu', W_constraint=maxnorm(3)))
model.add(Dense(1, activation='sigmoid'))

epochs = 100
learning_rate = 0.001
decay = learning_rate / epochs
sgd = SGD(lr=learning_rate, decay=decay, momentum=0.9, nesterov=True)
rms = RMSprop(lr=learning_rate, decay=decay)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
#early_stopping = EarlyStopping(monitor='acc', min_delt=0.01, patience=2, mode='max')
history = model.fit_generator(
    train_data_generator,
    samples_per_epoch=256,
    nb_epoch=100,
# callbacks=[early_stopping],
    validation_data=test_data_generator,
    nb_val_samples=2
)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#model.load_weights(filepath=filepath+"/shadowdetector.h5")
file = open(filepath+"/shadowdetector.h5", 'a')
model.save_weights(filepath=filepath+"/shadowdetector.h5")
file.close()

img = Image.open("data/train/nonshadows/04822305_257_1025_513_1281.jpg")
img = img.resize((150, 150), PIL.Image.ANTIALIAS)
img.save("data/train/nonshadows/04822305_257_1025_513_1281.jpg")
# i = cv2.imread("data/train/shadows/test/nonshadows/04822305_257_1025_513_1281.jpg")
# cv2.putText(i, "Shadow Detected", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# cv2.imshow("Object", i)
# cv2.waitKey(100000)
testImage = load_img("data/test/nonshadows/04822305_257_1025_513_1281.jpg")
x = img_to_array(testImage)
x = x.reshape((1,)+x.shape)
class_prediction = model.predict_classes(x)
# prediction = model.predict_generator(test_images, val_samples=4)
print(class_prediction)
get_layer_output = k.function([model.layers[0].input], model.layers[7].output)
layer_output = get_layer_output([x])[0]
print(layer_output)
