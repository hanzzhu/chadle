"""
Application stub
"""
from Halcon_Python.SettingVariables import SettingVariables
import numpy
from Halcon_Python.GUI import GUI
from Halcon_Python.makeCM import make_confusion_matrix, divide_chunks, figure
import tkinter as tk
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

from Halcon_Python.makeCM import make_confusion_matrix


def initialize():
    """

    Initialise Halcon

    """


    # path where dl_training_PK.hdl and dl_visulaization_PK.hdl files are located




def do_stuff():
    engine = ha.HDevEngine()
    engine.set_procedure_path('C:/MVTec/Halcon-20.11-Progress/procedures')
    engine.set_procedure_path(
        'C:/Users/930415/Desktop/Halcon DL library files')
    program = ha.HDevProgram('C:/Users/930415/Desktop/DL_train_CL_seagate.hdev')
    aug_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'augment_prepare'))
    preprocess_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'prepare_for_training'))
    training_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'train_dl_model_CE'))
    evaluation_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'Evaluation'))
    # do whatever you need to do
    aug_call.set_input_control_param_by_name('AugmentationPercentage', 0)
    aug_call.set_input_control_param_by_name('Rotation', 0)
    aug_call.set_input_control_param_by_name('Mirror', 'c')
    aug_call.set_input_control_param_by_name('BrightnessVariation', 0)
    aug_call.set_input_control_param_by_name('BrightnessVariationSpot', 0)
    aug_call.set_input_control_param_by_name('CropPercentage', 'off')
    aug_call.set_input_control_param_by_name('CropPixel', 'off')
    aug_call.set_input_control_param_by_name('RotationRange', 0)
    aug_call.set_input_control_param_by_name('IgnoreDirection', 'false')
    aug_call.set_input_control_param_by_name('ClassIDsNoOrientationExist', 'false')
    aug_call.set_input_control_param_by_name('ClassIDsNoOrientation', [])

    aug_call.execute()

    GenParamName_augment = aug_call.get_output_control_param_by_name('GenParamName_augment')
    GenParamValue_augment = aug_call.get_output_control_param_by_name('GenParamValue_augment')

    preprocess_call.set_input_control_param_by_name('RawImageBaseFolder', 'C:/Users/930415/Desktop/HK/animal')
    preprocess_call.set_input_control_param_by_name('ExampleDataDir', 'C:/Users/930415/Desktop/HK/Split')
    preprocess_call.set_input_control_param_by_name('BestModelBaseName',
                                                    os.path.join('C:/Users/930415/Desktop/HK/model', 'best_dl_model_classification'))
    preprocess_call.set_input_control_param_by_name('FinalModelBaseName',
                                                    os.path.join('C:/Users/930415/Desktop/HK/model', 'final_dl_model_classification'))
    preprocess_call.set_input_control_param_by_name('ImageWidth', 100)
    preprocess_call.set_input_control_param_by_name('ImageHeight', 100)
    preprocess_call.set_input_control_param_by_name('ImageNumChannels', 3)
    preprocess_call.set_input_control_param_by_name('ModelFileName', 'pretrained_dl_' + 'classifier_enhanced' + '.hdl')
    preprocess_call.set_input_control_param_by_name('BatchSize', 1)
    preprocess_call.set_input_control_param_by_name('InitialLearningRate', 0.1)
    preprocess_call.set_input_control_param_by_name('Momentum', 0.9)
    preprocess_call.set_input_control_param_by_name('DLDeviceType', 'cpu')
    preprocess_call.set_input_control_param_by_name('NumEpochs', 1)
    preprocess_call.set_input_control_param_by_name('ChangeLearningRateEpochs', [])
    preprocess_call.set_input_control_param_by_name('lr_change', [])
    preprocess_call.set_input_control_param_by_name('WeightPrior', 0.1)
    preprocess_call.set_input_control_param_by_name('EvaluationIntervalEpochs', 1)
    preprocess_call.set_input_control_param_by_name('Class_Penalty', [0,0])
    preprocess_call.set_input_control_param_by_name('GenParamName_augment', GenParamName_augment)
    preprocess_call.set_input_control_param_by_name('GenParamValue_augment', GenParamValue_augment)

    preprocess_call.execute()

    return ("Pre-process is done!~")

