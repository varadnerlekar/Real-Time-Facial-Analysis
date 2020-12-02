import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

def len_data(filename):
    anger_data = os.listdir(filename+'/anger')
    contempt_data = os.listdir(filename+'/contempt')
    disgust_data = os.listdir(filename+'/disgust')
    fear_data = os.listdir(filename+'/fear')
    happy_data = os.listdir(filename+'/happy')
    sadness_data = os.listdir(filename+'/sadness')
    surprise_data = os.listdir(filename+'/surprise')

    value = []
    for data in [anger_data, contempt_data, disgust_data, fear_data, happy_data, sadness_data, surprise_data]:
        value.append(len(data))
        
    return sum(value)

filename = 'CK+48'
print('Total Images in set : ' + str(len_data(filename)))

import cv2

def load_images_from_folder(folder):
    images = []
    
    folder1 = folder + '\\anger'
    for filename in os.listdir(folder1)[:20]:
        img = cv2.imread(os.path.join(os.getcwd(),folder1,filename))
        if img is not None:
            images.append(img)
            
    folder2 = folder + '\\contempt'
    for filename in os.listdir(folder1)[:20]:
        img = cv2.imread(os.path.join(os.getcwd(), folder2, filename))
        if img is not None:
            images.append(img)
            
    folder3 = folder + '\\disgust'
    for filename in os.listdir(folder1)[:20]:
        img = cv2.imread(os.path.join(os.getcwd(), folder3, filename))
        if img is not None:
            images.append(img)

    folder4 = folder + '\\fear'
    for filename in os.listdir(folder1)[:20]:
        img = cv2.imread(os.path.join(os.getcwd(), folder4, filename))
        if img is not None:
            images.append(img)

    folder5 = folder + '\\happy'
    for filename in os.listdir(folder1)[:20]:
        img = cv2.imread(os.path.join(os.getcwd(), folder5, filename))
        if img is not None:
            images.append(img)

    folder6 = folder + '\\sadness'
    for filename in os.listdir(folder1)[:20]:
        img = cv2.imread(os.path.join(os.getcwd(), folder6, filename))
        if img is not None:
            images.append(img)

    folder7 = folder + '\\surprise'
    for filename in os.listdir(folder1)[:20]:
        img = cv2.imread(os.path.join(os.getcwd(), folder7, filename))
        if img is not None:
            images.append(img)
            
    return images

images = load_images_from_folder('CK+48')


def plot_data(filename):
    anger_data = os.listdir(filename+'/anger')
    contempt_data = os.listdir(filename+'/contempt')
    disgust_data = os.listdir(filename+'/disgust')
    fear_data = os.listdir(filename+'/fear')
    happy_data = os.listdir(filename+'/happy')
    sadness_data = os.listdir(filename+'/sadness')
    surprise_data = os.listdir(filename+'/surprise')

    value = []
    for data in [anger_data,contempt_data,disgust_data,fear_data, happy_data, sadness_data, surprise_data]:
        value.append(len(data))
    
    sns.barplot(['angry','comtempt','disgust','fear','happy','sad','surprise'],value, palette = 'plasma')

filename = 'CK+48'
plot_data(filename)


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    )

train_generator = train_datagen.flow_from_directory(filename,
                                                    class_mode='categorical',
                                                    target_size=(48, 48))

class FinalCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.99):
            self.model.stop_training=True
callbacks = FinalCallback()


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(7, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

history = model.fit_generator(train_generator,
                              epochs=40,
                              verbose=1,callbacks = [callbacks])

print('Model has reached 99% accuracy. Training has stopped.')

model.save('FacialModel.h5')