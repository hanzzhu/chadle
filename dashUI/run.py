import os
import halcon as ha
# import thread
import logging
import threading
import time
import tkinter as Tk
from tkinter import messagebox

running = True  # Global flag


def pre_process(ProjectName):
    RootDir = 'C:/Users/930415/Desktop/Chadle_Project'
    ProjectDict = ['Animals', 'NTBW Image Analytics']
    if ProjectName in ProjectDict:
        ProjectDir = RootDir + '/' + ProjectName

        ModelDir = ProjectDir + '/Model'
        ModelFileName = 'pretrained_dl_classifier_enhanced.hdl'
        # Path to the image directory.
        HalconImageDir = ProjectDir + '/Image'

        SplitDir = ProjectDir + '/Split'

        ExampleDataDir = SplitDir + '/dldataset_100x100'
        DLDatasetFileName = ExampleDataDir + '/dl_dataset.hdict'

        DLPreprocessParamFileName = ExampleDataDir + '/dl_preprocess_param.hdict'
        BestModelBaseName = ModelDir + '/best_dl_model_classification'
        FinalModelBaseName = ModelDir + '/final_dl_model_classification'
        program = ha.HDevProgram('C:/Users/930415/Desktop/DL_train_CL_seagate.hdev')
        aug_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'augment_prepare'))
        preprocess_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'prepare_for_training'))
        # training_call = ha.HDevProcedureCall(ha.HDevProcedure.load_external('train_dl_model_PK'))
        # evaluation_call = ha.HDevProcedureCall(ha.HDevProcedure.load_external('Evaluation'))
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

        preprocess_call.set_input_control_param_by_name('RawImageBaseFolder', HalconImageDir)
        preprocess_call.set_input_control_param_by_name('ExampleDataDir', SplitDir)
        preprocess_call.set_input_control_param_by_name('BestModelBaseName',
                                                        os.path.join(ModelDir,
                                                                     'best_dl_model_classification'))
        preprocess_call.set_input_control_param_by_name('FinalModelBaseName',
                                                        os.path.join(ModelDir,
                                                                     'final_dl_model_classification'))
        preprocess_call.set_input_control_param_by_name('ImageWidth', 100)
        preprocess_call.set_input_control_param_by_name('ImageHeight', 100)
        preprocess_call.set_input_control_param_by_name('ImageNumChannels', 3)
        preprocess_call.set_input_control_param_by_name('ModelFileName',
                                                        ModelFileName)
        preprocess_call.set_input_control_param_by_name('BatchSize', 1)
        preprocess_call.set_input_control_param_by_name('InitialLearningRate', 0.1)
        preprocess_call.set_input_control_param_by_name('Momentum', 0.9)
        preprocess_call.set_input_control_param_by_name('DLDeviceType', 'cpu')
        preprocess_call.set_input_control_param_by_name('NumEpochs', 1)
        preprocess_call.set_input_control_param_by_name('ChangeLearningRateEpochs', [])
        preprocess_call.set_input_control_param_by_name('lr_change', [])
        preprocess_call.set_input_control_param_by_name('WeightPrior', 0.1)
        preprocess_call.set_input_control_param_by_name('EvaluationIntervalEpochs', 1)
        preprocess_call.set_input_control_param_by_name('Class_Penalty', [0, 0])
        preprocess_call.set_input_control_param_by_name('GenParamName_augment', GenParamName_augment)
        preprocess_call.set_input_control_param_by_name('GenParamValue_augment', GenParamValue_augment)
        preprocess_call.execute()

        DLModelHandle = preprocess_call.get_output_control_param_by_name('DLModelHandle')
        DLDataset = preprocess_call.get_output_control_param_by_name('DLDataset')
        TrainParam = preprocess_call.get_output_control_param_by_name('TrainParam')

        return DLModelHandle, DLDataset, TrainParam


def training(DLDataset, DLModelHandle, TrainParam):
    program = ha.HDevProgram('C:/Users/930415/Desktop/DL_train_CL_seagate.hdev')
    training_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'train_dl_model_PK'))
    # proc_training = ha.HDevProcedure.load_external('train_dl_model_CE')
    # proc_call = ha.HDevProcedureCall(proc_training)

    training_call.set_input_control_param_by_name('DLModelHandle', DLModelHandle)
    training_call.set_input_control_param_by_name('DLDataset', DLDataset)
    training_call.set_input_control_param_by_name('TrainParam', TrainParam)
    training_call.set_input_control_param_by_name('StartEpoch', 0)

    training_call.execute()


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
