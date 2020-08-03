import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras.utils import plot_model
from plot_confusion_matrix import plot_confusion_matrix

# Function that generates noise
def noise():
    x = np.random.randint(4, high = 32 - 4)
    y = np.random.randint(4, high = 32 - 4)
    
    noise = np.random.random((32, 32, 3))*0.2
    noise[x:x+4, y:y+4, ...] = np.random.randint(0, 2, size = 3)

    return noise

# Function for generating the dataset
def dataset_generator(batch_size = 16):
    img_shape = (32, 32, 3)
    object_size = 8 

    while True:
        x_train = np.zeros((batch_size, img_shape[0], img_shape[1], img_shape[2]))
        # The classes are represented by the colors
        y_train = np.zeros((batch_size, 6))

        for b in range(batch_size):

            selector = np.random.randint(0, high = 6, size = 1)

            x = np.random.randint(object_size, high = img_shape[0] - object_size)
            y = np.random.randint(object_size, high = img_shape[1] - object_size)
            
            # Create and object randomly (random position and random class)
            if selector == 1:
                x_train[b, x:x+object_size, y:y+object_size, 0] = 0.6
                y_train[b, 1] = 1.0
            elif selector == 2:
                x_train[b, x:x+object_size, y:y+object_size, 1] = 0.6
                y_train[b, 2] = 1.0
            elif selector == 3:
                x_train[b, x:x+object_size, y:y+object_size, 2] = 0.6
                y_train[b, 3] = 1.0
            elif selector == 4:
                x_train[b, x:x+object_size, y:y+object_size, 0] = 0.5
                x_train[b, x:x+object_size, y:y+object_size, 1] = 0.5
                y_train[b, 4] = 1.0
            elif selector == 5:
                x_train[b, x:x+object_size, y:y+object_size, 0] = 0.5
                x_train[b, x:x+object_size, y:y+object_size, 2] = 0.5
                y_train[b, 5] = 1.0
            else:
                x_train[b, ...] = 0
                y_train[b, 0] = 1.0
            
            # Add noise to the batches
            x_train[b, ...] = x_train[b, ...] + noise()
            x_train[b, ...] = np.clip(x_train[b, ...], 0.0, 1.0)
        
        yield x_train, y_train

# Create the model
def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Dense(512, activation = 'relu'))
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
    steps_per_epoch=1000//16,
    validation_data=dataset_generator(8),
    validation_steps=100//8,
    epochs = 10
)

# Test the model
gen = iter(dataset_generator(512))
x_train, y_train = next(gen)
pred = model.predict(x_train)


fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (12, 12))
counter = 0
for i in range(3):
    for j in range(3):
        axes[i,j].imshow(x_train[counter,...], cmap = 'gray')
        axes[i,j].set_title('Real: [%1.1f %1.1f %1.1f %1.1f %1.1f %1.1f]\nPred: [%1.1f %1.1f %1.1f %1.1f %1.1f %1.1f]'  
                            % (y_train[counter, 0], y_train[counter, 1], y_train[counter, 2], y_train[counter, 3],
                               y_train[counter, 4], y_train[counter, 5], pred[counter,0], pred[counter, 1],
                               pred[counter, 2], pred[counter, 3], pred[counter, 4], pred[counter, 5]))
        axes[i,j].axis(False)
        counter = counter + 1

plt.show()

# Plot confusion matrix
def getLabels(one_hot_encoded):
    l = np.argmax(one_hot_encoded, axis=1)
    labels = []

    for i in range(one_hot_encoded.shape[0]):
        if l[i] == 0:
            labels.append('None')
        elif l[i] == 1:
            labels.append('Red')
        elif l[i] == 2:
            labels.append('Green')
        elif l[i] == 3:
            labels.append('Blue')
        elif l[i] == 4:
            labels.append('Yellow')
        elif l[i] == 5:
            labels.append('Fuchsia')

    return labels


from sklearn.metrics import confusion_matrix

labels = ['None', 'Red', 'Green', 'Blue', 'Yellow', 'Fuchsia']

pl = getLabels(pred)
cl = getLabels(y_train)

cm = confusion_matrix(pl, cl, labels)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=['None', 'Red', 'Green', 'Blue', 'Yellow', 'Fuchsia'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=['None', 'Red', 'Green', 'Blue', 'Yellow', 'Fuchsia'], normalize=True,
                      title='Confusion matrix, without normalization')


