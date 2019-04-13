import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

image_width, image_height, batch_size = 150, 150, 32

def predict_images(file):
    model = load_model(file)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    images = []
    for folder in os.listdir('chest_xray/val'):
        if(folder == '.DS_Store'):
            continue
        for img in os.listdir('chest_xray/val/' + folder):
            if(img == '.DS_Store'):
                continue
            img = image.load_img('chest_xray/val/' + folder + '/' + img, target_size=(image_width, image_height))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            images.append(img)

    images = np.vstack(images)
    predictions = model.predict_classes(images, batch_size=batch_size)
    for object in predictions:
        if object == 0:
            print("No tumor")
        elif object == 1:
            print("Tumor")

if __name__ == "__main__":
    predict_images('pneumonia.h5')
