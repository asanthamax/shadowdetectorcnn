import keras
import numpy
import cv2
import PIL
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from keras.engine import Layer, merge, Merge, Input, Model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from matplotlib import pyplot
from keras.layers import Dense, Flatten, MaxPooling2D, Reshape, Permute, Activation, Dropout, LeakyReLU
from keras.optimizers import RMSprop, SGD, Adam
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, UpSampling2D, Deconvolution2D, Cropping2D
from keras import backend as k

from objectpreprocessor import custom_loss, LossHistory, TestAcc, generate_batch_data

k.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)
train_directory = 'data/train/train'
test_directory = 'data/test/test'
filepath = "data/models"


input_img = Input(shape=(3, 150, 150))
cnn = Convolution2D(16, 2, 2, border_mode='same')(input_img)
cnn = LeakyReLU(alpha=0.1)(cnn)
cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cnn)

cnn = Convolution2D(32, 2, 2, border_mode='same')(cnn)
cnn = LeakyReLU(alpha=0.1)(cnn)
cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cnn)

cnn = Convolution2D(64, 2, 2, border_mode='same')(cnn)
cnn = LeakyReLU(alpha=0.1)(cnn)
cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cnn)

cnn = Convolution2D(128, 2, 2, border_mode='same')(cnn)
cnn = LeakyReLU(alpha=0.1)(cnn)
cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cnn)

cnn = Convolution2D(256, 2, 2, border_mode='same')(cnn)
cnn = LeakyReLU(alpha=0.1)(cnn)
cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cnn)

cnn = Convolution2D(512, 2, 2, border_mode='same')(cnn)
cnn = LeakyReLU(alpha=0.1)(cnn)
cnn = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cnn)

cnn = Convolution2D(1024, 2, 2, border_mode='same')(cnn)
cnn = LeakyReLU(alpha=0.1)(cnn)


cnn = Convolution2D(1024, 2, 2, border_mode='same')(cnn)
cnn = LeakyReLU(alpha=0.1)(cnn)

cnn = Convolution2D(1024, 2, 2, border_mode='same')(cnn)
cnn = LeakyReLU(alpha=0.1)(cnn)

cnn = Flatten()(cnn)
cnn = Dense(4096)(cnn)
cnn = LeakyReLU(alpha=0.1)(cnn)
cnn = Dense(7*7*25, activation='sigmoid')(cnn)

model = Model(input_img, cnn)
epochs = 100
learning_rate = 0.001
decay = learning_rate / epochs
sgd = SGD(lr=learning_rate, decay=decay, momentum=0.9, nesterov=True)
rms = RMSprop(lr=learning_rate, decay=decay)
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss=custom_loss, optimizer=sgd, metrics=['accuracy'])
print(model.summary())

#history = LossHistory()
#testAcc = TestAcc()

#history = model.fit_generator(
#    generate_batch_data('data/VOC2012', 'data/VOC2012/ImageSets/Segmentation/trainval.txt', 16, sample_number=2913),
#    samples_per_epoch=5826,
#    nb_epoch=40,
#    callbacks=[history, testAcc]
#)

file = open(filepath+"/recognizer.h5", 'a')
#model.save_weights(filepath=filepath+"/recognizer.h5")
model.load_weights(filepath=filepath+"/recognizer.h5")
img = Image.open("data/VOC2012/JPEGImages/2007_000733.jpg")
img = img.resize((150, 150), PIL.Image.ANTIALIAS)
img.save("data/VOC2012/JPEGImages/2007_000733.jpg")
# i = cv2.imread("data/train/shadows/test/nonshadows/04822305_257_1025_513_1281.jpg")
# cv2.putText(i, "Shadow Detected", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# cv2.imshow("Object", i)
# cv2.waitKey(100000)
testImage = load_img("data/VOC2012/JPEGImages/2007_000733.jpg")
x = img_to_array(testImage)
x = x.reshape((1,)+x.shape)
labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
img = Image.open("data/VOC2012/JPEGImages/2007_000733.jpg")
img = img.resize((150, 150), PIL.Image.ANTIALIAS)
img.save("data/VOC2012/JPEGImages/2007_000733.jpg")
drawImage = Image.open("data/VOC2012/JPEGImages/2007_000733.jpg")
drawer = ImageDraw.Draw(drawImage)
prediction = model.predict(x)
prediction = prediction[0]
final_output = []
for i in range(49):
    thresh = 0.3
    out = prediction[i*25: (i+1)* 25]
    if(out[24] > thresh):

        row = i/7
        col = i/7
        centerx = 64 * col + 64 * prediction[0]
        centery = 64 * row + 64 * prediction[1]
        h = prediction[3] * prediction[3]
        h = h * 150
        w = prediction[4] * prediction[4]
        w = w * 150
        left = centerx - w/2.0
        right = centerx + w/2.0
        up = centery - h/2.0
        down = centery + h/2.0


        if(left < 0):left = 0
        if(right > 150):right = 150
        if(up < 0):up = 0
        if(down > 150):down = 150

        drawer.rectangle([left, up, right, down], outline='red')
        if(left == 0):
            xcoordinate = 0
        else:
            xcoordinate = left - 1
        if(up == 0):
            ycoordinate = 0
        else:
            ycoordinate = up - 1

        final_output.append(left)
        final_output.append(right)
        final_output.append(up)
        final_output.append(down)
        final_output.append(numpy.argmax(prediction[2:24]))
        drawer.text((xcoordinate, ycoordinate), labels[numpy.argmax(prediction[2:24])])
drawImage.save("data/VOC2012/JPEGImages/prediction1.jpg")
i = cv2.imread("data/VOC2012/JPEGImages/prediction1.jpg")
cv2.imshow("Object", i)
cv2.waitKey(100000)
print(model.predict(x))
print(final_output)

