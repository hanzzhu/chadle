# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:02:13 2021

@author: ET-GTX2
"""

import os
import halcon as ha
# import thread
import logging
import threading
import time
import tkinter as Tk
from tkinter import messagebox

running = True  # Global flag

###### Input/output directories ######
"""""
ExampleDataDir = 'C:/Users/930415/Desktop/NTBW_Image Analytics/training_findinghub'
ModelFileName = 'pretrained_dl_classifier_compact.hdl'
DataDirectory = ExampleDataDir + '/dldataset_fipg_960x1024'
DLDatasetFileName = DataDirectory + '/dl_dataset.hdict'
DLPreprocessParamFileName = DataDirectory + '/dl_preprocess_param.hdict'
BestModelBaseName = 'C:/Users/930415/Desktop/NTBW_Image Analytics/models_hub' + '/best_dl_model_classification'
FinalModelBaseName = 'C:/Users/930415/Desktop/NTBW_Image Analytics/models_hub' + '/final_dl_model_classification'
"""
ImageWidth = str(960)
ImageHeight = str(1024)

# Main training directory
RootDir = 'C:/Users/930415/Desktop/NTBW_Image Analytics/'
MainDir = RootDir + 'training_findinghub'
# Path to the image directory.
HalconImageDir = 'C:/Users/930415/Desktop/NTBW_Image Analytics/TapeRouting_Pictures'

ExampleDataDir = MainDir
ModelFileName = MainDir + '/pretrained_dl_model_detection.hdl'
DataDirectory = ExampleDataDir + '/dldataset_fipg_' + ImageWidth + 'x' + ImageHeight
DLDatasetFileName = DataDirectory + '/dl_dataset.hdict'
DLPreprocessParamFileName = MainDir + '/dl_preprocess_param.hdict'
BestModelBaseName = ExampleDataDir + '/best_dl_model_classification'
FinalModelBaseName = ExampleDataDir + '/final_dl_model_classification'


###### Parameter settings ######
BatchSize = 10
InitialLearningRate = 0.005
Momentum = 0.9
NumEpochs = 20
StartEpoch = 0
EvaluationIntervalEpochs = 1
ChangeLearningRateEpochs = [4, 8, 12]
lr_change = [0.1, 0.01, 0.001]
WeightPrior = 1
Class_Penalty = []  # Each class should be assigned with a penalty value

###### Augmentation parameter settings ######
# The percentage of the images that are to be augmented.
AugmentationPercentage = 20  # Expects integer value in the range [0, 100]

# Step size for possible rotations.
Rotation = 0  # expects values in the range [-180, 180]

# Allowed mirroring types are coded by 'r' (row), 'c' (column).
Mirror = 'off'  # expects 'r' or 'c' or 'rc'

# Absolute brightness change can vary in the range [-value, +value]
BrightnessVariation = 0

# The absolute brightness peak of a randomly positioned spot can vary in the range [-value, +value]
BrightnessVariationSpot = 0

###### Special augmentation parameters for Classification module ######
# Fraction of image length and width that remains after cropping (in %).
CropPercentage = 'off'  # Expects 'off' or value in %

# Image length and width that remains after cropping (in pixel).
CropPixel = 'off'  # Expects 'off' or integer value

# Step range for rotations with step size 1.
RotationRange = 10

###### Special augmentation parameters for Object Detection module ######
# In case of a detection model of instance_type 'rectangle2': Use directions of instances within bounding boxes.
IgnoreDirection = 'false'  # Expects true or false

# In case of a detection model of instance_type 'rectangle2': Class IDs without orientation.
ClassIDsNoOrientationExist = 'false'  # Expect true or false
ClassIDsNoOrientation = []  # Default []


def setup_hdev_engine():
    """Setup HDevEngine by setting procedure search paths."""
    source_dir = 'E:/Customer evaluation/Seagate/'
    hdev_source_dir = source_dir

    engine = ha.HDevEngine()
    engine.set_procedure_path('C:/Program Files/MVTec/HALCON-20.11-Progress/procedures')

    engine.set_procedure_path(
        'C:/Users/930415/Desktop/Halcon DL library files')  # path where dl_training_PK.hdl and dl_visulaization_PK.hdl files are located

    # engine.set_procedure_path('E:/Customer evaluation/Seagate/HDev Engine_Python/dl_training_PK.hdpl')
    # engine.set_procedure_path('E:/Customer evaluation/Seagate/HDev Engine_Python/dl_visualization_PK.hdpl')
    return hdev_source_dir


def augment_prepare(proc_name_augment):
    proc = ha.HDevProcedure.load_external(proc_name_augment)
    # proc = ha.HDevProcedureCall('train_dl_model_PK')
    proc_call = ha.HDevProcedureCall(proc)

    proc_call.set_input_control_param_by_name('AugmentationPercentage', AugmentationPercentage)
    proc_call.set_input_control_param_by_name('Rotation', Rotation)
    proc_call.set_input_control_param_by_name('Mirror', Mirror)
    proc_call.set_input_control_param_by_name('BrightnessVariation', BrightnessVariation)
    proc_call.set_input_control_param_by_name('BrightnessVariationSpot', BrightnessVariationSpot)
    proc_call.set_input_control_param_by_name('CropPercentage', CropPercentage)
    proc_call.set_input_control_param_by_name('CropPixel', CropPixel)
    proc_call.set_input_control_param_by_name('RotationRange', RotationRange)
    proc_call.set_input_control_param_by_name('IgnoreDirection', IgnoreDirection)
    proc_call.set_input_control_param_by_name('ClassIDsNoOrientationExist', ClassIDsNoOrientationExist)
    proc_call.set_input_control_param_by_name('ClassIDsNoOrientation', ClassIDsNoOrientation)

    proc_call.execute()

    GenParamName_augment = proc_call.get_output_control_param_by_name('GenParamName_augment')
    GenParamValue_augment = proc_call.get_output_control_param_by_name('GenParamValue_augment')

    return GenParamName_augment, GenParamValue_augment


def prep_for_training(GenParamName_augment, GenParamValue_augment, proc_name):
    """Execute procedure for image acquisition."""
    proc = ha.HDevProcedure.load_external(proc_name)
    proc_call = ha.HDevProcedureCall(proc)

    proc_call.set_input_control_param_by_name('ExampleDataDir', ExampleDataDir)
    proc_call.set_input_control_param_by_name('ModelFileName', ModelFileName)
    # proc_call.set_input_control_param_by_name('DataDirectory', DataDirectory)
    proc_call.set_input_control_param_by_name('DLDatasetFileName', DLDatasetFileName)
    proc_call.set_input_control_param_by_name('DLPreprocessParamFileName', DLPreprocessParamFileName)
    proc_call.set_input_control_param_by_name('BestModelBaseName', BestModelBaseName)
    proc_call.set_input_control_param_by_name('FinalModelBaseName', FinalModelBaseName)
    proc_call.set_input_control_param_by_name('BatchSize', BatchSize)
    proc_call.set_input_control_param_by_name('InitialLearningRate', InitialLearningRate)
    proc_call.set_input_control_param_by_name('Momentum', Momentum)
    proc_call.set_input_control_param_by_name('NumEpochs', NumEpochs)
    proc_call.set_input_control_param_by_name('EvaluationIntervalEpochs', EvaluationIntervalEpochs)
    proc_call.set_input_control_param_by_name('ChangeLearningRateEpochs', ChangeLearningRateEpochs)
    proc_call.set_input_control_param_by_name('lr_change', lr_change)
    proc_call.set_input_control_param_by_name('WeightPrior', WeightPrior)
    proc_call.set_input_control_param_by_name('GenParamName_augment', GenParamName_augment)
    proc_call.set_input_control_param_by_name('GenParamValue_augment', GenParamValue_augment)
    proc_call.set_input_control_param_by_name('Class_Penalty', Class_Penalty)

    proc_call.execute()

    DLModelHandle = proc_call.get_output_control_param_by_name('DLModelHandle')
    DLDataset = proc_call.get_output_control_param_by_name('DLDataset')
    TrainParam = proc_call.get_output_control_param_by_name('TrainParam')

    return DLModelHandle, DLDataset, TrainParam


def control_execution():
    idx = 0  # loop index

    def stop():
        """Enable scanning by setting the global flag to True."""
        global running
        running = True
        FileHandle = ha.open_file('mutex.dat', 'output')

        ha.fwrite_string(FileHandle, 1)
        # close_file(FileHandle)

    def resume():
        """Stop scanning by setting the global flag to False."""
        global running
        running = False
        FileHandle = ha.open_file('mutex.dat', 'output')
        ha.fwrite_string(FileHandle, 0)
        # close_file(FileHandle)

    def doSomething():
        root.destroy()

    def on_exit():
        """When you click to exit, this function is called"""
        if messagebox.askyesno("Exit", "Do you want to quit the application?"):
            root.destroy()
            FileHandle = ha.open_file('mutex.dat', 'output')
            ha.fwrite_string(FileHandle, 3)

    root = Tk.Tk()
    root.title("Contol execution")
    root.protocol('WM_DELETE_WINDOW', on_exit)  # root is your root window
    # root.geometry('200x250 + 400 + 300')

    start = Tk.Button(root, text="Stop", command=stop)
    stop = Tk.Button(root, text="Resume", command=resume)

    start.grid()
    stop.grid()

    # open_file(), fwrite_line() and close_file() to modify the 'mutex.dat'

    while True:
        if idx % 500 == 0:
            root.update()

        if running:
            idx += 1

        else:
            idx += 1


# def doSomething():
#     # check if saving
#     # if not:
#     root.destroy()


def training(DLDataset, DLModelHandle, TrainParam):
    proc_training = ha.HDevProcedure.load_external('train_dl_model_CE')
    proc_call = ha.HDevProcedureCall(proc_training)

    proc_call.set_input_control_param_by_name('DLModelHandle', DLModelHandle)
    proc_call.set_input_control_param_by_name('DLDataset', DLDataset)
    proc_call.set_input_control_param_by_name('TrainParam', TrainParam)
    proc_call.set_input_control_param_by_name('StartEpoch', StartEpoch)

    proc_call.execute()


if __name__ == '__main__':
    hdev_source_dir = setup_hdev_engine()
    x = threading.Thread(target=control_execution)
    x.start()

    FileHandle = ha.open_file('mutex.dat', 'output')
    # Augmentation
    proc_name_augment = 'augment_prepare'
    proc_augment_preparation = augment_prepare(proc_name_augment)

    GenParamName_augment = proc_augment_preparation[0][0]
    GenParamValue_augment = proc_augment_preparation[1][0]

    # Preparation
    proc_name = 'prepare_for_training'
    proc_preparation = prep_for_training(GenParamName_augment, GenParamValue_augment, proc_name)

    DLModelHandle = proc_preparation[0][0]
    DLDataset = proc_preparation[1][0]
    TrainParam = proc_preparation[2][0]

    # Training
    proc_training = training(DLDataset, DLModelHandle, TrainParam)
