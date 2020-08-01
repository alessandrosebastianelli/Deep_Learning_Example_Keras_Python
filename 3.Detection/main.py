import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import backend as K

from tensorflow.keras.utils import plot_model

def dataset_generator(batch_size = 16):
    img_shape = (32, 32, 3)
    

    while True:
        x_train = np.zeros((batch_size, img_shape[0], img_shape[1], img_shape[2]))
        y_train = np.zeros((batch_size, 4))

        for b in range(batch_size):

            object_width = int(np.random.randint(4, high = 12, size = 1))
            object_height = int(np.random.randint(4, high = 12, size = 1)) 
            #object_height = object_width

            x = np.random.randint(object_width + 1, high = img_shape[0] - object_width - 1)
            y = np.random.randint(object_height + 1, high = img_shape[0] - object_height - 1)

            x_train[b, x:x+object_width, y:y+object_height, 1] = 1
            #y_train[b, 0] = 1.0

            y_train[b, 0] = x
            y_train[b, 1] = y
            y_train[b, 2] = x + object_width
            y_train[b, 3] = y + object_height
        
        yield x_train, y_train

def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64,  activation = 'relu'))
    model.add(Dense(32,  activation = 'relu'))
    model.add(Dense(4))

    model.compile(loss='mse', optimizer = 'adam')

    print(model.summary)
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model

def bounding_box(coordinates):
    box = np.zeros((32, 32, 3))
    box[:, int(round(coordinates[1], 1)) -1 , 0] = 1
    box[:, int(round(coordinates[3], 1)) + 1, 0] = 1
    box[int(round(coordinates[0], 1)) - 1, :, 0] = 1
    box[int(round(coordinates[2], 1)) + 1, :, 0] = 1

    return box
    

model = build_model()

history = model.fit_generator(
    dataset_generator(16),
    steps_per_epoch=4000//16,
    validation_data=dataset_generator(8),
    validation_steps=400//8,
    epochs = 100
)


gen = iter(dataset_generator(16))
x_train, y_train = next(gen)
pred = model.predict(x_train)

print(y_train)
print(pred)


fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (12, 12))

counter = 0
for i in range(3):
    for j in range(3):
        b = bounding_box(pred[counter, ...])
        axes[i,j].imshow(x_train[counter,...] + b)
        axes[i,j].set_title('Real: [%1.1f %1.1f %1.1f %1.1f]\nPred: [%1.1f %1.1f %1.1f %1.1f]'  
                            % (y_train[counter,0], y_train[counter, 1], y_train[counter, 2], y_train[counter, 3]
                               , pred[counter,0], pred[counter, 1], pred[counter, 2], pred[counter, 3]))
        axes[i,j].axis(False)
        counter = counter + 1

plt.show()
