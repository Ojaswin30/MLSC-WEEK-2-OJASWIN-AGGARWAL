import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
new_model = load_model(os.path.join('models','happysadmodel.h5'))

img = cv2.imread('Happy sad image recognizer//happy test.jpg')
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

resize = tf.image.resize(img, (256,256))
# plt.imshow(resize.numpy().astype(int))
# plt.show()
yhat_new = new_model.predict(np.expand_dims(resize/255, 0))

print(yhat_new)

if yhat_new > 0.5:
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')

