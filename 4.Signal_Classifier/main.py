import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras.utils import plot_model
from plot_confusion_matrix import plot_confusion_matrix


# Function for generating the dataset
def dataset_generator(batch_size = 16):
    x_in = np.linspace(-3*np.pi, 3*np.pi, 3000)

    while True:
        x_train = np.zeros((batch_size, 3000))
        # The classes are represented by the colors
        y_train = np.zeros((batch_size, 6))

        for b in range(batch_size):

            selector = np.random.randint(0, high = 6, size = 1)
 
            
            # Create and object randomly (random position and random class)
            if selector == 1:
                x_train[b, ...] = 0.8*np.sin(x_in) - 0.3*np.random.normal(0, 1, size = 3000)
                y_train[b, 1] = 1.0
            elif selector == 2:
                x_train[b, ...] = 0.7*np.sin(x_in) + 0.4*np.random.normal(0, 1, size = 3000)
                y_train[b, 2] = 1.0
            elif selector == 3:
                x_train[b, ...] = 0.6*np.sin(x_in) - 0.5*np.random.normal(0, 1, size = 3000)
                y_train[b, 3] = 1.0
            elif selector == 4:
                x_train[b, ...] = 0.5*np.sin(x_in) + 0.6*np.random.normal(0, 1, size = 3000)
                y_train[b, 4] = 1.0
            elif selector == 5:
                x_train[b, ...] = 0.4*np.sin(x_in) - 0.7*np.random.normal(0, 1, size = 3000)
                y_train[b, 5] = 1.0
            else:
                x_train[b, ...] = 0.3*np.sin(x_in) + 0.8*np.random.normal(0, 1, size = 3000)
                y_train[b, 0] = 1.0
              
        yield x_train, y_train

# Create the model
def build_model():
    model = Sequential()
    model.add(Dense(512, input_dim = 3000, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(6, activation = 'softmax'))

    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

    print(model.summary)
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model


# Create the model
model = build_model()

# Train the model
history = model.fit_generator(
    dataset_generator(16),
    steps_per_epoch=3000//16,
    validation_data=dataset_generator(8),
    validation_steps=500//8,
    epochs = 20
)

# Test the model
gen = iter(dataset_generator(512))
x_train, y_train = next(gen)
pred = model.predict(x_train)


fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (12, 12))
counter = 0
for i in range(3):
    for j in range(3):
        axes[i,j].plot(x_train[counter,...])
        axes[i,j].set_title('Real: [%1.1f %1.1f %1.1f %1.1f %1.1f %1.1f]\nPred: [%1.1f %1.1f %1.1f %1.1f %1.1f %1.1f]'  
                            % (y_train[counter, 0], y_train[counter, 1], y_train[counter, 2], y_train[counter, 3],
                               y_train[counter, 4], y_train[counter, 5], pred[counter,0], pred[counter, 1],
                               pred[counter, 2], pred[counter, 3], pred[counter, 4], pred[counter, 5]))
        axes[i,j].grid()
        axes[i,j].set_ylim(-2, 2)
        counter = counter + 1

plt.show()

# Plot confusion matrix
def getLabels(one_hot_encoded):
    l = np.argmax(one_hot_encoded, axis=1)
    labels = []

    for i in range(one_hot_encoded.shape[0]):
        if l[i] == 0:
            labels.append('0.3')
        elif l[i] == 1:
            labels.append('0.8')
        elif l[i] == 2:
            labels.append('0.7')
        elif l[i] == 3:
            labels.append('0.6')
        elif l[i] == 4:
            labels.append('0.5')
        elif l[i] == 5:
            labels.append('0.4')

    return labels


from sklearn.metrics import confusion_matrix

labels = ['0.3', '0.8', '0.7', '0.6', '0.5', '0.4']

pl = getLabels(pred)
cl = getLabels(y_train)

cm = confusion_matrix(pl, cl, labels)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=['0.3', '0.8', '0.7', '0.6', '0.5', '0.4'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=['0.3', '0.8', '0.7', '0.6', '0.5', '0.4'], normalize=True,
                      title='Confusion matrix, without normalization')


