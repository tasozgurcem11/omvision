import dataloader
import metrics
import preprocessing
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


from preprocessing import *
from dataloader import *
from submission import *
from metrics import *



ckpt = r"C:\Users\ilkay\Documents\Cem-Berkan\checkpoint"
checkpoint_path = r'C:\Users\ilkay\Documents\Cem-Berkan\checkpoint'
model_dir = r"C:\Users\ilkay\Documents\Cem-Berkan\models\{date}.h5" 
image_dir = r"C:\Users\ilkay\Documents\Cem-Berkan\Data\stage_2_train"
path = r"C:\Users\ilkay\Documents\Cem-Berkan\Data\stage_2_test"
data_path = r"C:\Users\ilkay\Documents\Cem-Berkan\Data\DATA.csv"
submission_path = r'C:\Users\ilkay\Documents\Cem-Berkan\submission.csv'
sample_submission_path = r"C:\Users\ilkay\Documents\Cem-Berkan\Data\stage_2_sample_submission.csv"
train_csv_path = r"C:\Users\ilkay\Documents\Cem-Berkan\Data\stage_2_train.csv"


params = {'dim':(224,224,3),
         'batch_size':32,
         'n_classes':6,
         'n_channels':0,
         'shuffle':True}
epochs = 4
# Generators

df = get_dataframe(train_csv_path)#path is argument

partition,labels = get_partition_labels(df,frac = 1)

training_generator = DataGenerator(preprocessing.partition['train'], labels, **params)
validation_generator = DataGenerator(preprocessing.partition['validation'], labels, **params)


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

epochs = 20
# Train model on dataset
model.fit(training_generator,
         validation_data=validation_generator,
         epochs=epochs,
          workers = 2, 
          use_multiprocessing= True,
         callbacks = my_callbacks)


from datetime import date
date = date.today
model.save(model_dir)