import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from tensorflow import  keras
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

import pandas as pd
from tensorflow import keras
import tqdm
import pydicom

from tqdm import tqdm

import tensorflow as tf 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

physical_devices =  tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)


ckpt = r"C:\Users\ilkay\Documents\Cem-Berkan\checkpoint"
checkpoint_path = r'C:\Users\ilkay\Documents\Cem-Berkan\checkpoint'
model_dir = r"C:\Users\ilkay\Documents\Cem-Berkan\models\{date}.h5" 
image_dir = r"C:\Users\ilkay\Documents\Cem-Berkan\Data\stage_2_train"
path = r"C:\Users\ilkay\Documents\Cem-Berkan\Data\stage_2_test"
data_path = r"C:\Users\ilkay\Documents\Cem-Berkan\Data\DATA.csv"
submission_path = r'C:\Users\ilkay\Documents\Cem-Berkan\submission.csv'
sample_submission_path = r"C:\Users\ilkay\Documents\Cem-Berkan\Data\stage_2_sample_submission.csv"
train_csv_path = r"C:\Users\ilkay\Documents\Cem-Berkan\Data\stage_2_train.csv"

print("Log1: Import successful")

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
                image_dir = image_dir
                X[i,] = _read(image_dir+ID+'.dcm',self.dim)
                #print(X)
                # Store class
                y[i] = self.labels[ID]
                #print(y[i])
            return X,y
        
        else: # test phase
            for i, ID in enumerate(list_IDs_temp):
                path = path
                X[i,] = _read(path+ID+'.dcm',self.dim)
            
            return X



def weighted_log_loss(y_true, y_pred):
    """
    Can be used as the loss function in model.compile()
    ---------------------------------------------------
    """
    
    class_weights =  tf.constant([2., 1., 1., 1., 1., 1.])
    
    eps = tf.keras.backend.epsilon()
    
    
    y_pred = tf.clip_by_value(y_pred, eps, 1.0-eps)

    out = -(         y_true  * tf.math.log(      y_pred) * class_weights
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred) * class_weights)
    
    return tf.reduce_mean(out, axis=-1)


####
def weighted_log_loss_V2(y_true, y_pred):
    """
    Can be used as the loss function in model.compile()
    ---------------------------------------------------
    """
    
    class_weights =  tf.constant([2., 1., 1., 1., 1., 1.])
    
    eps = tf.keras.backend.epsilon()
    
    
    y_pred = tf.clip_by_value(y_pred, eps, 1.0-eps)

    out = -(         y_true  * tf.math.log(      y_pred) * class_weights
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred) * class_weights)
    
    return tf.reduce_mean(out, axis=-1)
####

def _normalized_weighted_average(arr, weights=None):
    """
    A simple Keras implementation that mimics that of 
    numpy.average(), specifically for this competition
    """
    
    if weights is not None:
        scl = K.sum(weights)
        weights = K.expand_dims(weights, axis=1)
        return K.sum(K.dot(arr, weights), axis=1) / scl
    return K.mean(arr, axis=1)


def weighted_loss(y_true, y_pred):
    """
    Will be used as the metric in model.compile()
    ---------------------------------------------
    
    Similar to the custom loss function 'weighted_log_loss()' above
    but with normalized weights, which should be very similar 
    to the official competition metric:
        https://www.kaggle.com/kambarakun/lb-probe-weights-n-of-positives-scoring
    and hence:
        sklearn.metrics.log_loss with sample weights
    """
    
    class_weights = tf.constant([2., 1., 1., 1., 1., 1.])
    
    eps = tf.keras.backend.epsilon()
    
    y_pred = tf.clip_by_value(y_pred, eps, 1.0-eps)

    loss = -(        y_true  * tf.math.log(      y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    
    loss_samples = _normalized_weighted_average(loss, class_weights)
    
    return tf.reduce_mean(loss_samples)


def weighted_log_loss_metric(trues, preds):
    """
    Will be used to calculate the log loss 
    of the validation set in PredictionCheckpoint()
    ------------------------------------------
    """
    class_weights = [2., 1., 1., 1., 1., 1.]
    
    epsilon = 1e-7
    
    preds = np.clip(preds, epsilon, 1-epsilon)
    loss = trues * np.log(preds) + (1 - trues) * np.log(1 - preds)
    loss_samples = np.average(loss, axis=1, weights=class_weights)

    return - loss_samples.mean()
'''
def log_loss(y_pred,y_true):
    #y_pred = np.clip(y_pred, 1e-7, 1.0-1e-7)
    ones = np.where(y_true)
    zeros = np.where(y_true == 0)
    likelihood = np.prod(y_pred[ones]) *  np.prod(y_pred[zeros])
    logloss = -1 * np.log(likelihood)
    return logloss
'''

def log_loss(y_pred,y_true):
    """
    Will be used to calculate the log loss 
    of the validation set in metrics
    ------------------------------------------
    """
    eps = 1e-15
    
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    
    ones = tf.where(y_true)
    zeros = tf.where(y_true == 0)
    
    likelihood = tf.math.reduce_prod(tf.gather(y_pred,ones)) *  tf.math.reduce_prod(1 - tf.gather(y_pred,zeros))
    logloss = -1 * tf.math.log(likelihood)
    return logloss


def get_partition_labels(df,frac = 0.25):
    partition = dict()
    labels = dict()
    df = df.sample(frac = frac, random_state= 1)
    for i in trange(len(df)):
        id_ = df.Image.iloc[i]
        #print('id = ',id_,type(id_))
        label = df.iloc[i,1:7].to_numpy(dtype = 'int32')
        #print('label =',label)
        labels[id_] = label
    
    training = df.sample(frac = 0.6,random_state= 1)
    validation = df.drop(training.index, axis = 0)
    test = validation.sample(frac = 0.5,random_state= 1)
    validation = validation.drop(test.index, axis = 0) 
    partition['train'] = list(training.Image)
    partition['validation'] = list(validation.Image)
    partition['test'] = list(test.Image)
    return partition,labels


def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

def window_image(dcm, window_center, window_width):
    
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)

    return img

def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)
    
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)

    return 

def window_with_correction(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img


def window_testing(img, window):
    brain_img = window(img, 40, 80)
    subdural_img = window(img, 80, 200)
    soft_img = window(img, 40, 380)
    
    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)

    return bsb_img





def get_dataframe(train_csv_path):
    df1 = pd.read_csv(train_csv_path)
    df1["Image"] = df1["ID"].str.slice(stop=12)
    df1["Diagnosis"] = df1["ID"].str.slice(start=13)

    duplicates_to_remove = [
            56346, 56347, 56348, 56349,
            56350, 56351, 1171830, 1171831,
            1171832, 1171833, 1171834, 1171835,
            3705312, 3705313, 3705314, 3705315,
            3705316, 3705317, 3842478, 3842479,
            3842480, 3842481, 3842482, 3842483]
    df1 = df1.drop(index=duplicates_to_remove)
    df1 = df1.reset_index(drop=True)
    df1 = df1.loc[:, [ "Diagnosis", "Image","Label"]]
    df1 = df1.pivot(index = 'Image', columns = 'Diagnosis', values = 'Label')
    DATA = []
    columns = ['PatientID','SOPInstanceUID','SeriesInstanceUID','StudyInstanceUID','z']
    
    for dcm in tqdm(df1.index):
        dicom = pydicom.dcmread(data_path +dcm+'.dcm')
        PatientID = dicom.PatientID
        SOPInstanceUID = dicom.SOPInstanceUID
        SeriesInstanceUID = dicom.SeriesInstanceUID
        StudyInstanceUID = dicom.StudyInstanceUID
        z_value = dicom.ImagePositionPatient[-1]
        DATA.append([PatientID,SOPInstanceUID,SeriesInstanceUID,StudyInstanceUID,z_value])
    new_df = pd.DataFrame(DATA,index = df1.index, columns = columns)
    overall = pd.concat([df1,new_df],axis = 1)
    overall.to_csv(data_path)
    return overall

def get_test_dcms():
    path = sample_submission_path
    df1 = pd.read_csv(path)
    df1["Image"] = df1["ID"].str.slice(stop=12)
    df1["Diagnosis"] = df1["ID"].str.slice(start=13)
    df1 = df1.loc[:, [ "Diagnosis", "Image","Label"]]
    df1 = df1.pivot(index = 'Image', columns = 'Diagnosis', values = 'Label')
    return list(df1.index)

def get_pred(path = checkpoint_path):
    params = {'dim':(224,224,3),
         'batch_size':16,
         'n_classes':6,
         'n_channels':0,
         'shuffle':True}
    dcms = get_test_dcms() #121232 element
    test_generator = DataGenerator(dcms,labels = None,**params)
    model = tf.keras.models.load_model( path, compile = False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.weighted_loss,metrics.log_loss,keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(num_thresholds=200, curve='ROC',summation_method='interpolation', multi_label=True, label_weights=[2, 1, 1, 1, 1, 1])])
    #model = model.load_weights(path)
    preds = model.predict(test_generator,verbose = 1,use_multiprocessing = True, workers = 4)
    return preds

def get_submission(preds):
    dcms = get_test_dcms()
    deneme = pd.DataFrame(preds, columns=[ 'epidural', 'intraparenchymal', 'intraventricular','subarachnoid', 'subdural','any'], index=dcms)
    values = deneme.stack().values
    path = sample_submission_path
    submission = pd.read_csv(path)
    submission.drop_duplicates()
    submission.Label = values
    #print(submission.head())
    submission.to_csv(submission_path,index = None)
    return submission


params = {'dim':(224,224,3),
         'batch_size':32,
         'n_classes':6,
         'n_channels':0,
         'shuffle':True}
epochs = 4
# Generators

df = get_dataframe(train_csv_path)#path is argument

partition,labels = get_partition_labels(df,frac = 1)

print("Log2: Partition successful")


training_generator = DataGenerator(preprocessing.partition['train'], labels, **params)
validation_generator = DataGenerator(preprocessing.partition['validation'], labels, **params)

print("Log3: DataGeneration successful")

# Design model
model = Sequential([
    tf.keras.applications.ResNet101(
    include_top=False,
    weights="imagenet",
    input_shape=params['dim'],
    pooling='max'),
    Flatten(),
    Dense(30,activation = 'relu'),
    Dense(6,activation = 'sigmoid')
]) 


my_callbacks = [
    ModelCheckpoint(filepath= ckpt, monitor = 'val_loss' ,save_best_only=True,save_weights_only=False,verbose = 1),
    ReduceLROnPlateau(monitor= 'val_loss', factor=0.1, patience= 2, verbose=1,mode='auto', min_delta=0.0001)]

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.weighted_loss,metrics.log_loss, keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(num_thresholds=200, curve='ROC',summation_method='interpolation', multi_label=True, label_weights=[2, 1, 1, 1, 1, 1])])

print("Log4: Model Compile successful")

epochs = 20
# Train model on dataset
model.fit(training_generator,
         validation_data=validation_generator,
         epochs=epochs,
          workers = 2, 
          use_multiprocessing= True,
         callbacks = my_callbacks)

print("Log5: Model Fit successful")


from datetime import date
date = date.today
model.save(model_dir)


