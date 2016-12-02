import json

from datetime import timedelta, datetime
from keras.layers import Convolution2D, Flatten, Dense
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop
import numpy as np

from experiencereplay import ExperienceReplay


class RLModel:

    def __init__(self):

        self.model = Sequential()
        self.experience = ExperienceReplay(max_memory=1000)
        self.time_period = timedelta(minutes=15) + datetime.now()
        self.isroundcomplete = self.time_period < datetime.now()

    def initialize_network(self):

        self.model.add(
            Convolution2D(32, 8, 8, activation='relu', border_mode='same', subsample=(4, 4), input_shape=(3, 200, 200)))
        self.model.add(Convolution2D(64, 4, 4, activation='relu', border_mode='same', subsample=(2, 2)))
        self.model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(4, activation='linear'))

        learning_rate = 0.001
        epochs = 10000
        decay = learning_rate / epochs
        rms = RMSprop(lr=learning_rate, decay=decay)
        self.model.compile(loss='mse', optimizer=rms)

    def train_model(self, img_path):

        epsilon = 0.1
        num_outputs = 4
        epoch = 1000
        positive_action_count = 0
        for e in range(epoch):

            loss = 0.
            input_t = self.experience.get_initial_input(self.model, img_path)
            while not self.isroundcomplete:
                input_nw = input_t

                if np.random.rand() <= epsilon:
                    action = np.random.rand(0, num_outputs, size=1)
                else:
                    qval = self.model.predict(input_nw)
                    action = np.argmax(qval[0])

                input_t, reward, self.isroundcomplete = self.apply_action(action)
                if reward == 1:
                    positive_action_count += 1

                inputs, target = self.experience.get_train_batch(self.model, img_path)
                loss += self.model.train_on_batch(inputs, target)[0]
            print("Epoch {:03d}/1000 | Loss {:.4f} | Positive action count {}".format(e, loss, positive_action_count))
        self.model.save_weights('data/models/rlmodel.h5', overwrite=True)
        with open('data/models/rlmodel.json', 'w') as outfile:
            json.dump(self.model.to_json(), outfile)

    def apply_action(self, action):

        input_t = self.experience.get_initial_input(self.model, "")
        status = 0
        if status == 1:
            reward = 1
        elif status == 2:
            reward = -1
        else:
            reward = 0
        self.isroundcomplete = self.time_period < datetime.now()
        return input_t, reward, self.isroundcomplete

    def test_model(self, stop=False):

        with open('data/models/rlmodel.h5', 'r') as jfile:
            self.model = model_from_json(json.load(jfile))
        self.model.load_weights('data/models/rlmodel.h5')
        learning_rate = 0.001
        epochs = 10000
        decay = learning_rate / epochs
        rms = RMSprop(lr=learning_rate, decay=decay)
        self.model.compile(loss='mse', optimizer=rms)
        input_t = self.experience.get_initial_input(self.model, '')
        while not stop:
            input_nw = input_t
            qval = self.model.predict(input_nw)
            action = np.argmax(qval[0])
            reward = 1
            input_t = self.experience.get_initial_input(self.model, "")