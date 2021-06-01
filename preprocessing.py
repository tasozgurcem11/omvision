import numpy as np
import pydicom
from tqdm import trange
from __init__ import *


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



import tqdm
import pandas as pd
import pydicom

from tqdm import tqdm

def get_dataframe(train_csv_path,data_path):
    #path = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train.csv'
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
    #path = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train/'
    
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

