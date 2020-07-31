import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras.utils import plot_model

def dataset_generator(batch_size = 16):
    img_shape = (32, 32, 3)
    obgject_size = 8
    

    while True:
        x_train = np.zeros((batch_size, img_shape[0], img_shape[1], img_shape[2]))
        y_train = np.zeros((batch_size, 4))

        for b in range(batch_size):

            selector = np.random.randint(0, high = 4, size = 1)

            x = np.random.randint(obgject_size, high = img_shape[0] - obgject_size)
            y = np.random.randint(obgject_size, high = img_shape[0] - obgject_size)

            if selector == 1:
                x_train[b, x:x+obgject_size, y:y+obgject_size, 0] = 1
                y_train[b, 1] = 1.0
            elif selector == 2:
                x_train[b, x:x+obgject_size, y:y+obgject_size, 1] = 1
                y_train[b, 2] = 1.0
            elif selector == 3:
                x_train[b, x:x+obgject_size, y:y+obgject_size, 2] = 1
                y_train[b, 3] = 1.0
            else:
                x_train[b, ...] = 0
                y_train[b, 0] = 1.0
        
        yield x_train, y_train

def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(4, activation = 'softmax'))

    model.compile(loss='binary_crossentropy', optimizer = 'adam')

    print(model.summary)
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model



model = build_model()

history = model.fit_generator(
    dataset_generator(16),
    steps_per_epoch=1000//16,
    validation_data=dataset_generator(8),
    validation_steps=100//8,
    epochs = 10
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
        axes[i,j].imshow(x_train[counter,...], cmap = 'gray')
        axes[i,j].set_title('Real: [%1.1f %1.1f %1.1f %1.1f]\nPred: [%1.1f %1.1f %1.1f %1.1f]'  
                            % (y_train[counter,0], y_train[counter, 1], y_train[counter, 2], y_train[counter, 3], 
                               pred[counter,0], pred[counter, 1], pred[counter, 2], pred[counter, 3]))
        axes[i,j].axis(False)
        counter = counter + 1

plt.show()
