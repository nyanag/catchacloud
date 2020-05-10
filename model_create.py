import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.common.set_image_dim_ordering('th')
from matplotlib.image import imread
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
arm = np.load('data/arm.npy')
arm = arm.reshape((arm.shape[0], 1, 28, 28))

apple = np.load('data/apple.npy')
apple = apple.reshape((apple.shape[0], 1, 28, 28))
cat = np.load('data/cat.npy')
cat = cat.reshape((cat.shape[0], 1, 28, 28))
car = np.load('data/car.npy')
car = car.reshape((car.shape[0], 1, 28, 28))
dog = np.load('data/dog.npy')
dog = dog.reshape((dog.shape[0], 1, 28, 28))
horse = np.load('data/horse.npy')
horse = horse.reshape((horse.shape[0], 1, 28, 28))
face = np.load('data/face.npy')
face = face.reshape((face.shape[0], 1, 28, 28))
banana = np.load('data/banana.npy')
banana=banana.reshape((banana.shape[0], 1, 28, 28))
bus = np.load('data/bus.npy')
bus = bus.reshape((bus.shape[0], 1, 28, 28))
bat = np.load('data/bat.npy')
bat = bat.reshape((bat.shape[0], 1, 28, 28))

X_data = np.concatenate([arm,apple,cat,car,dog,horse,face,banana,bus,bat], axis = 0)
arm_y = np.full(len(arm), 0)
apple_y=np.full(len(apple),1)
cat_y = np.full(len(cat), 2)
car_y = np.full(len(car), 3)
dog_y=np.full(len(dog), 4)
horse_y=np.full(len(horse), 5)
face_y=np.full(len(face), 6)
banana_y=np.full(len(banana), 7)
bus_y=np.full(len(bus), 8)
bat_y=np.full(len(bat), 9)

Y_data = np.concatenate([arm_y,apple_y,cat_y,car_y,dog_y,horse_y,face_y,banana_y,bus_y,bat_y], axis = 0)
Y_data = np_utils.to_categorical(Y_data)


from sklearn.utils import shuffle
index = np.arange(len(X_data))
print(index[0])
index, Y_data = shuffle(index,Y_data,random_state=0)
index_train, index_test, y_train, y_test = train_test_split(index,Y_data,test_size=0.2,random_state=0)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32,shuffle=False):
        'Initialization'
       
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        # Find list of IDs
        list_IDs_temp = self.list_IDs[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X = self.__data_generation(list_IDs_temp)
        y = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size,1,28,28))
        

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = X_data[ID]

        X=X.astype(np.float32)
        X=X/255

        return X


def cnn_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu',data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = cnn_model()


training_generator = DataGenerator(index_train, y_train)
test_generator= DataGenerator(index_test, y_test)

model.fit_generator(generator=training_generator,
                    validation_data=test_generator,
                    use_multiprocessing=False)

'''import cv2

img = cv2.imread('image_path')
bilateral = cv2.bilateralFilter(img, 15, 75, 75)
cv2.imshow('filter',bilateral)
img = bilateral
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#img = 0.5*high_thresh
edges = cv2.Canny(img,8,8)
cv2.imwrite('load.jpg',edges)

image = cv2.imread('load.jpg')
lol = image
lol = cv2.resize(image,(512,512))
image = cv2.resize(image, (28, 28))

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image.reshape((-1, 28, 28,1)) 
image = np.moveaxis(image, -1, 0)
digit = model.predict_classes(image)
lit = {
    0:'arm',1:'apple',2:'cat',3:'car',4:'dog',5:'horse',6:'face',7:'banana',8:'bus',9:'bat'
}
print(digit[0])
cv2.imshow('lit',lol)
cv2.waitKey(10000)
cv2.destroyAllWindows()'''

with open('custom_cnn.json', 'w') as f:
    f.write(model.to_json())

model.save_weights('model_weights.json')
