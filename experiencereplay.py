from PIL import Image
import numpy as np


class ExperienceReplay:

    def __init__(self, max_memory=100000, discount=0.9):

        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, status, isRoundComplete):

        self.memory.append([status, isRoundComplete])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_initial_input(self,model, imgPath, resizeWidth=200, resizeHeight=200):

        memory_length = len(self.memory)
        num_outputs = model.output_shape[-1]
        image = Image.open(imgPath)
        image = image.resize((resizeWidth, resizeHeight), Image.ANTIALIAS)
        image_array = np.array(image)
        image_array = np.rollaxis(image_array,2,0)
        image_array = image_array/255.0
        image_array = image_array * 2.0 - 1.0
        inputs = np.zeros((min(memory_length, num_outputs), image_array))
        return inputs

    def get_train_batch(self, model, img, resizeWidth=200, resizeHeight=200, batchSize=32):

        memory_length = len(self.memory)
        num_outputs = model.output_shape[-1]
        image = Image.open(img)
        image = image.resize((resizeWidth, resizeHeight), Image.ANTIALIAS)
        image_array = np.array(image)
        image_array = np.rollaxis(image_array,2,0)
        image_array = image_array/255.0
        image_array = image_array * 2.0 - 1.0

        inputs = np.zeros((min(memory_length, num_outputs), image_array))
        targets = np.zeros((inputs.shape[0], num_outputs))

        for i, index in enumerate(np.random.randint(0, memory_length, size=inputs.shape[0])):

            state_t, action_t, reward_t, state_tn = self.memory[index][0]
            testRoundComplete = self.memory[index][1]
            inputs[i : i+1] = state_t
            targets[i] = model.predict(state_t)[0]
            Qvalue = np.argmax(model.predict(state_tn)[0])
            if testRoundComplete:
                targets[i, action_t] = reward_t
            else:
                targets[i, action_t] = reward_t + self.discount * Qvalue

        return inputs, targets