import plaidml.keras
plaidml.keras.install_backend()

import keras
import cv2

CATEGORIES = ["Dog", "Cat"]

def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = keras.models.load_model("64x3-CNN.model")

for i in range(1, 6):
    prediction = model.predict([prepare(f"Prediction Data/dog{i}.jpg")])
    print(CATEGORIES[int(prediction[0][0])])
