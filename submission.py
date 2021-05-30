
from dataloader import DataGenerator
import tensorflow as tf
import pandas as pd
import metrics
from tensorflow import keras

def get_test_dcms():
    path = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv'
    df1 = pd.read_csv(path)
    df1["Image"] = df1["ID"].str.slice(stop=12)
    df1["Diagnosis"] = df1["ID"].str.slice(start=13)
    df1 = df1.loc[:, [ "Diagnosis", "Image","Label"]]
    df1 = df1.pivot(index = 'Image', columns = 'Diagnosis', values = 'Label')
    return list(df1.index)

def get_pred(path = './checkpoint'):
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
    path = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv'
    submission = pd.read_csv(path)
    submission.drop_duplicates()
    submission.Label = values
    #print(submission.head())
    submission.to_csv('./submission.csv',index = None)
    return submission