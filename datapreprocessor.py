from keras.preprocessing.image import ImageDataGenerator


class ImagePreprocessor:
    img_width = 150
    img_height = 150
    train_directory = 'data/train'
    test_directory = 'data/test'

    def preProcessTrainImageSet(self):

        train_datagen = ImageDataGenerator(

            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        train_data_generator = train_datagen.flow_from_directory(

            self.train_directory,
            target_size=(self.img_width,self.img_height),
            batch_size=32,
            class_mode='binary'
        )
        return train_data_generator

    def preProcessTestImageSet(self):

        test_datagen = ImageDataGenerator(

            rescale=1./255
        )

        test_data_generator = test_datagen.flow_from_directory(

            self.test_directory,
            target_size=(self.img_width,self.img_height),
            batch_size=32,
            class_mode='binary'
        )

        return test_data_generator
