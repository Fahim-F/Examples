from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pdb
model = ResNet50(weights='imagenet')

img_path = './figures/ILSVRC2012_test_00098768.JPEG'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
#list of imagenet classes https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
print('Predicted:', decode_predictions(preds, top=3)[0])
