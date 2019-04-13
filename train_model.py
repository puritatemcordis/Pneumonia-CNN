import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

image_width, image_height = 150, 150

def create_model(p, input_shape=(64, 64, 1)):
    model = Sequential()

    model.add(Convolution2D(32, (3,3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Flatten())

    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(p))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(p/2))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train(batch_size=32, epochs=10):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory('chest_xray/train', target_size=(image_width, image_height), batch_size=batch_size, class_mode='binary')

    test_set = test_datagen.flow_from_directory('chest_xray/test', target_size=(image_width, image_height), batch_size=batch_size, class_mode='binary')

    model = create_model(p=0.5, input_shape=(image_width, image_height, 3))
    training_history = model.fit_generator(training_set, steps_per_epoch=(training_set.n/training_set.batch_size), epochs=epochs, validation_data=test_set, validation_steps=(test_set.n/test_set.batch_size))

    model.save("pneumonia.h5")
    model.save_weights("pneumonia_weight.h5")
    print("Saved model on disk")

    plot_training_history(training_history, 'acc', 'val_acc')

def plot_training_history(training_history, train_acc, test_acc):
    plt.plot(training_history.history[train_acc])
    plt.plot(training_history.history[test_acc])
    plt.title('Training History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('pneumonia_training.png')
    plt.show()

def main():
    train(batch_size=32, epochs=50)

if __name__ == "__main__":
    main()
