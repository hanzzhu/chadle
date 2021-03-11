import tkinter as tk
from tkinter import filedialog

from GUI import GUI
from Halcon_Python.SettingVariables import SettingVariables
from Halcon_Python.SettingVariables import SettingVariables
import numpy
from Halcon_Python.GUI import GUI
from makeCM import make_confusion_matrix, divide_chunks, figure

from tkinter.ttk import Progressbar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from tkinter import *
import threading
import os
from mttkinter import *
import halcon as ha

from makeCM import make_confusion_matrix

class ObjectDetection(GUI):
    def __init__(self, parent, controller):
        GUI.__init__(self, parent)

        styles = SettingVariables.frame_styles
        dir_color = SettingVariables.dirColor
        backgroundColor = SettingVariables.backgroundColor
        # initialise frames
        # frameSettings, frame Parameters,frame_Inspection_graphical, frameTop
        frame_Parameters = tk.LabelFrame(self, styles, text="Parameters")
        frame_Parameters.place(rely=0.07, relx=0.02, height=300, width=600)

        frame_Augmentation = tk.LabelFrame(self, styles, text="Augmentation")
        frame_Augmentation.place(rely=0.47, relx=0.02, height=300, width=600)

        frameSettings = tk.LabelFrame(self, styles, text="Settings")
        frameSettings.place(rely=0.07, relx=0.45, height=180, width=700)

        frame_Inspection_graphical = tk.LabelFrame(self, styles, text="Graphical Inspection")
        frame_Inspection_graphical.place(rely=0.3, relx=0.45, height=500, width=550)

        frame_Inspection_stats = tk.LabelFrame(self, styles, text="Statistics")
        frame_Inspection_stats.place(rely=0.3, relx=0.83, height=500, width=150)

        # frameTop = tk.LabelFrame(self, styles, )
        # frameTop.place(rely=0.01, relx=0.02, height=30, width=900)

        frameBot = tk.LabelFrame(self, styles, )
        frameBot.place(rely=0.97, relx=0, height=30, width=1440)

        ObjectDetection_label = tk.Label(self, text='Object Detection', font=('Verdana', 20, 'bold'),
                                        bg=backgroundColor,fg='dark red')
        ObjectDetection_label.place(rely=0, relx=0.1, height=50)

        ImageDirDirectory_path = StringVar()
        PreprocessDirDirectory_path = StringVar()
        ModelDirDirectory_path = StringVar()

        def ImageDirDirectory():
            path = filedialog.askdirectory(initialdir="/", title="Select folder", )
            ImageDirDirectory_path.set(path)
            ImageDirGotLabel.config(text=path)

        ImageDirGotLabel = tk.Label(frameSettings, width=47, font=('calibre', 10, 'bold'),
                                    bg=("%s" % dir_color))
        ImageDirGotLabel.grid(row=1, column=1)
        ImageDirButton = tk.Button(frameSettings, text="Image Directory", command=lambda: ImageDirDirectory())
        ImageDirButton.grid(row=1, column=0)

        def PreprocessDirDirectory():
            path = filedialog.askdirectory(initialdir="/", title="Select folder", )
            PreprocessDirDirectory_path.set(path)
            PreprocessDirGotLabel.config(text=path)

        PreprocessDirButton = tk.Button(frameSettings, text="Preprocess Directory",
                                        command=lambda: PreprocessDirDirectory())
        PreprocessDirButton.grid(row=2, column=0)
        PreprocessDirGotLabel = tk.Label(frameSettings, width=47, font=('calibre', 10, 'bold'), bg=dir_color)
        PreprocessDirGotLabel.grid(row=2, column=1)

        def ModelDirDirectory():
            path = filedialog.askdirectory(initialdir="/", title="Select folder", )
            ModelDirDirectory_path.set(path)
            ModelDirGotLabel.config(text=path)

        ModelDirButton = tk.Button(frameSettings, text="Model Directory", command=lambda: ModelDirDirectory())
        ModelDirButton.grid(row=3, column=0)
        ModelDirGotLabel = tk.Label(frameSettings, width=47, font=('calibre', 10, 'bold'), bg=dir_color)
        ModelDirGotLabel.grid(row=3, column=1)

        # Dropdown list for pretrained model
        PretrainedModelList = ["classifier_enhanced", "classifier_compact"]
        PretrainedModelList_variable = tk.StringVar()
        PretrainedModelList_variable.set(PretrainedModelList[0])
        dropList = tk.OptionMenu(frameSettings, PretrainedModelList_variable, *PretrainedModelList)
        dropList.config(width=20, font=('calibre', 10, 'bold'), bg=dir_color)
        dropList.grid(row=5, column=1)
        dropListLabel = tk.Label(frameSettings, text="Pretrained model: ", font=('calibre', 10, 'bold'),
                                 bg=backgroundColor)
        dropListLabel.grid(row=5, column=0)
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:52:50 2021

@author: ET-GTX2
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:52:50 2021

@author: ET-GTX2
"""

import os

import halcon as ha

###### Input/output directories ######
ImageWidth = str(512)
ImageHeight = str(320)

# Main training directory
RootDir = 'E:/Customer evaluation/Seagate/HDevEngine_Python_OD/'
MainDir = RootDir + 'detect_pills_data'
# Path to the image directory.
HalconImageDir = 'E:/MVTec/Halcon-20.11-Progress/examples/images/pill_bag'

ExampleDataDir = MainDir
ModelFileName = './detect_pills_data/pretrained_dl_model_detection.hdl'
DataDirectory = ExampleDataDir + '/dldataset_pill_bag_' + ImageWidth + 'x' + ImageHeight
DLDatasetFileName = DataDirectory + '/dl_dataset.hdict'
DLPreprocessParamFileName = DataDirectory + '/dl_preprocess_param.hdict'
BestModelBaseName = ExampleDataDir + '/best_dl_model_classification'
FinalModelBaseName = ExampleDataDir + '/final_dl_model_classification'

###### Parameter settings ######

BatchSize = 'maximum'
InitialLearningRate = 1e-3
Momentum = 0.9
NumEpochs = 3
StartEpoch = 0
EvaluationIntervalEpochs = 1
ChangeLearningRateEpochs = [4, 8, 12]

###### Advanced Parameter settings ######

WeightPrior = 0.0005

lr_change = [0.1, 0.01, 0.001]

Class_Penalty = [1.0, 1.0, 1.0]  # Each class should be assigned with a penalty value

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
# Not applicable for object detection
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
    engine.set_procedure_path('E:/MVTec/Halcon-20.11-Progress/procedures')

    engine.set_procedure_path(
        'E:/Customer evaluation/Seagate/HDevEngine_Python_OD/HDPL files')  # path where dl_training_PK.hdl and dl_visulaization_PK.hdl files are located

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


def training(DLDataset, DLModelHandle, TrainParam):
    proc_training = ha.HDevProcedure.load_external('train_dl_model_PK')
    proc_call = ha.HDevProcedureCall(proc_training)

    proc_call.set_input_control_param_by_name('DLModelHandle', DLModelHandle)
    proc_call.set_input_control_param_by_name('DLDataset', DLDataset)
    proc_call.set_input_control_param_by_name('TrainParam', TrainParam)
    proc_call.set_input_control_param_by_name('StartEpoch', StartEpoch)

    proc_call.execute()


if __name__ == '__main__':
    hdev_source_dir = setup_hdev_engine()

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







