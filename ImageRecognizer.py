import tensorflow as tf
import os

import cv2
import imghdr
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

data_dir= 'Happy sad image recognizer//Data'
# print(os.listdir(os.path.join(data_dir, 'happy')))

image_exts = ['jpeg','jpg','bmp','png']
# print(image_exts[2])

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))

tf.data.Dataset.list_files

import numpy as np

data = tf.keras.utils.image_dataset_from_directory('Happy sad image recognizer//Data')

data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

# print(len(batch))

# print(batch[1])

# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])

# plt.show()


# scaled = batch[0] / 255
# print(scaled.max())

# Preprocess data

data = data.map(lambda x,y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
# print(scaled_iterator.next()[0])


# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])
# plt.show()

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)







###########Deep model#############

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()

model.add(Conv2D(16, (3,3),1, activation = 'relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D((32), (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D((16), (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
# print(model.summary())




#####model training


logdir='Happy sad image recognizer//logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir)
hist = model.fit(train, epochs=20, validation_data = val, callbacks = [tensorboard_callback])
# print(hist.history)



# plot performance


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal',label='loss')
plt.plot(hist.history['val_loss'], color='orange',label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal',label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange',label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


####evaluate and testing


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    x,y = batch
    yhat = model.predict(x)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

# print(f'Precision:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

#  test

# img = cv2.imread('Happy sad image recognizer//happy test.jpg')
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()

# resize = tf.image.resize(img, (256,256))
# plt.imshow(resize.numpy().astype(int))
# plt.show()

# np.expand_dims(resize, 0)
# yhat = model.predict(np.expand_dims(resize/255, 0))

# print(yhat)

# if yhat > 0.5:
#     print(f'Predicted class is Sad')
# else:
#     print(f'Predicted class is Happy')


#  Save the model
from tensorflow.keras.models import load_model
model.save(os.path.join('models','happysadmodel.h5'))