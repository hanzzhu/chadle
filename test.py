# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:52:50 2021

@author: ET-GTX2
"""
import halcon as ha
import threading
import multiprocessing

engine = ha.HDevEngine()
engine.set_procedure_path('C:/Program Files/MVTec/HALCON-20.11-Progress/procedures')

engine.set_procedure_path(
        'C:/Users/930415/Desktop/Halcon DL library files')

proc = ha.HDevProcedure.load_external('augment_prepare')
augment_proc_call = ha.HDevProcedureCall(proc)
prep_for_training_call = ha.HDevProcedureCall(ha.HDevProcedure.load_external('prepare_for_training'))
training_call = ha.HDevProcedureCall(ha.HDevProcedure.load_external('train_dl_model_PK'))
###### Input/output directories ######
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

BatchSize = 1
InitialLearningRate = 1e-3
Momentum = 0.9
NumEpochs = 1
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
    # source_dir = 'E:/Customer evaluation/Seagate/'
    # hdev_source_dir = source_dir

    engine = ha.HDevEngine()
    engine.set_procedure_path('C:/Program Files/MVTec/HALCON-20.11-Progress/procedures')

    engine.set_procedure_path(
        'C:/Users/930415/Desktop/Halcon DL library files')  # path where dl_training_PK.hdl and dl_visulaization_PK.hdl files are located

    # engine.set_procedure_path('E:/Customer evaluation/Seagate/HDev Engine_Python/dl_training_PK.hdpl')
    # engine.set_procedure_path('E:/Customer evaluation/Seagate/HDev Engine_Python/dl_visualization_PK.hdpl')


def augment_prepare():
    augment_proc_call.set_input_control_param_by_name('AugmentationPercentage', AugmentationPercentage)
    augment_proc_call.set_input_control_param_by_name('Rotation', Rotation)
    augment_proc_call.set_input_control_param_by_name('Mirror', Mirror)
    augment_proc_call.set_input_control_param_by_name('BrightnessVariation', BrightnessVariation)
    augment_proc_call.set_input_control_param_by_name('BrightnessVariationSpot', BrightnessVariationSpot)
    augment_proc_call.set_input_control_param_by_name('CropPercentage', CropPercentage)
    augment_proc_call.set_input_control_param_by_name('CropPixel', CropPixel)
    augment_proc_call.set_input_control_param_by_name('RotationRange', RotationRange)
    augment_proc_call.set_input_control_param_by_name('IgnoreDirection', IgnoreDirection)
    augment_proc_call.set_input_control_param_by_name('ClassIDsNoOrientationExist', ClassIDsNoOrientationExist)
    augment_proc_call.set_input_control_param_by_name('ClassIDsNoOrientation', ClassIDsNoOrientation)

    augment_proc_call.execute()


def prep_for_training():
    """Execute procedure for image acquisition."""
    GenParamName_augment = augment_proc_call.get_output_control_param_by_name('GenParamName_augment')
    GenParamValue_augment = augment_proc_call.get_output_control_param_by_name('GenParamValue_augment')

    prep_for_training_call.set_input_control_param_by_name('ExampleDataDir', ExampleDataDir)
    prep_for_training_call.set_input_control_param_by_name('ModelFileName', ModelFileName)
    # proc_call.set_input_control_param_by_name('DataDirectory', DataDirectory)
    prep_for_training_call.set_input_control_param_by_name('DLDatasetFileName', DLDatasetFileName)
    prep_for_training_call.set_input_control_param_by_name('DLPreprocessParamFileName', DLPreprocessParamFileName)
    prep_for_training_call.set_input_control_param_by_name('BestModelBaseName', BestModelBaseName)
    prep_for_training_call.set_input_control_param_by_name('FinalModelBaseName', FinalModelBaseName)
    prep_for_training_call.set_input_control_param_by_name('BatchSize', BatchSize)
    prep_for_training_call.set_input_control_param_by_name('InitialLearningRate', InitialLearningRate)
    prep_for_training_call.set_input_control_param_by_name('Momentum', Momentum)
    prep_for_training_call.set_input_control_param_by_name('NumEpochs', NumEpochs)
    prep_for_training_call.set_input_control_param_by_name('EvaluationIntervalEpochs', EvaluationIntervalEpochs)
    prep_for_training_call.set_input_control_param_by_name('ChangeLearningRateEpochs', ChangeLearningRateEpochs)

    prep_for_training_call.set_input_control_param_by_name('lr_change', lr_change)
    prep_for_training_call.set_input_control_param_by_name('WeightPrior', WeightPrior)
    prep_for_training_call.set_input_control_param_by_name('GenParamName_augment', GenParamName_augment)
    prep_for_training_call.set_input_control_param_by_name('GenParamValue_augment', GenParamValue_augment)
    prep_for_training_call.set_input_control_param_by_name('Class_Penalty', Class_Penalty)

    prep_for_training_call.execute()


def training():
    DLDataset = prep_for_training_call.get_output_control_param_by_name('DLDataset')
    DLModelHandle = prep_for_training_call.get_output_control_param_by_name('DLModelHandle')
    TrainParam = prep_for_training_call.get_output_control_param_by_name('TrainParam')
    print(DLModelHandle,DLDataset,TrainParam)
    training_call.set_input_control_param_by_name('DLModelHandle', DLModelHandle)
    training_call.set_input_control_param_by_name('DLDataset', DLDataset)
    training_call.set_input_control_param_by_name('TrainParam', TrainParam)
    training_call.set_input_control_param_by_name('StartEpoch', StartEpoch)

    training_call.execute()


if __name__ == '__main__':
    setup_hdev_engine()

    # Augmentation
    augment_prepare()

    prep_for_training()

    # proc_preparation = prep_for_training(GenParamName_augment, GenParamValue_augment, proc_name)

    # Training

    threading.Thread(target=training).start()
