import sys
import numpy as np
import json
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

img = load_img(sys.argv[1], target_size=(224, 224))  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 300, 300)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

datagen = ImageDataGenerator()

for batch in datagen.flow(x, batch_size=1, save_format='jpg'):
    print(json.dumps({"images": batch.tolist(), "key": 0}))
    break
