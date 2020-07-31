import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.utils import plot_model


def dataset_generator(batch_size, time_size): 
    x_train = np.zeros((batch_size, time_size))
    y_train = np.zeros((batch_size, time_size))
    t = np.linspace(-3*np.pi, 3*np.pi, time_size)

    while True:
        
        for i in range(batch_size):
            x_train[i,...] = np.sin(t)
            y_train[i,...] = np.sin(t)
            for dt in range(time_size):
                error = np.random.normal(0, 1, size = 1)
                x_train[i, dt] = x_train[i, dt] + 0.2*error

        yield x_train, y_train

def build_model(time_size):
    model = Sequential()
    model.add(Dense(128, input_dim=time_size, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(time_size, activation='tanh'))
    model.compile(loss='mse', optimizer='adam')

    print(model.summary())
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model
   


time_size = 300
model = build_model(time_size)


history = model.fit_generator(
    dataset_generator(16, time_size),
    steps_per_epoch=1000//16,
    validation_data=dataset_generator(16, time_size),
    validation_steps=1000//16,
    epochs=15
)

#fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (5, 5))
#ax.plot(history.history['loss'], label = 'Training Loss')
#ax.plot(history.history['val_loss'], label = 'Validation Loss')

#ax.legend()
#plt.show()


x_train, y_train = next(iter(dataset_generator(10, time_size)))
pred = model.predict(x_train)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (10, 10))

t = np.linspace(-3*np.pi, 3*np.pi, time_size)
axes.plot(t, x_train[0,...], label = 'Additive noise')
axes.plot(t, y_train[0,...], label = 'Ground Truth')
axes.plot(t, pred[0,...], label = 'Model Prediction')

axes.legend()
axes.grid()

plt.show()



