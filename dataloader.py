import numpy as np
from tensorflow import  keras
from preprocessing import _read

#Data loader specialized

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,n_classes=10, shuffle=True):
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
        if self.labels is not None:
            X, y = self.__data_generation(list_IDs_temp)
            return X, y
        else:
            X = self.__data_generation(list_IDs_temp)
            return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim),dtype = 'float32') #,self.n_channels))
        if self.labels is not None: # training phase
            #Use self.n_classes instead of 6 
            y = np.empty((self.batch_size,6), dtype='float32')
            #print(y.shape)
            #print(X.shape)
            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                image_dir = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train/'
                X[i,] = _read(image_dir+ID+'.dcm',self.dim)
                #print(X)
                # Store class
                y[i] = self.labels[ID]
                #print(y[i])
            return X,y
        
        else: # test phase
            for i, ID in enumerate(list_IDs_temp):
                path = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_test/'
                X[i,] = _read(path+ID+'.dcm',self.dim)
            
            return X