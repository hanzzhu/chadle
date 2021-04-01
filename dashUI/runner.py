import os
import halcon as ha
import json


with open('parameters_json.txt') as json_file:
    ParameterDict = json.load(json_file)

def pre_process(ProjectName, Runtime, PretrainedModel, ImWidth, ImHeight, ImChannel,
                BatchSize, InitialLearningRate, Momentum, NumEpochs, ChangeLearningRateEpochs, lr_change, WeightPrior,
                class_penalty, AugmentationPercentage, Rotation, mirror, BrightnessVariation, BrightnessVariationSpot,
                CropPercentage, CropPixel, RotationRange, IgnoreDirection, ClassIDsNoOrientationExist,
                ClassIDsNoOrientation):
    RootDir = 'C:/Users/930415/Desktop/Chadle_Project'
    ProjectDict = ['Animals', 'NTBW Image Analytics']
    if ProjectName in ProjectDict:
        ProjectDir = RootDir + '/' + ProjectName

        ModelDir = ProjectDir + '/Model'
        ModelFileName = 'pretrained_dl_' + PretrainedModel + '.hdl'
        # Path to the image directory.
        HalconImageDir = ProjectDir + '/Image'

        SplitDir = ProjectDir + '/Split'

        ExampleDataDir = SplitDir + '/dldataset_' + str(ImWidth) + 'x' + str(ImHeight)
        DLDatasetFileName = ExampleDataDir + '/dl_dataset.hdict'

        DLPreprocessParamFileName = ExampleDataDir + '/dl_preprocess_param.hdict'
        BestModelBaseName = ModelDir + '/best_dl_model_classification'
        FinalModelBaseName = ModelDir + '/final_dl_model_classification'
        program = ha.HDevProgram('C:/Users/930415/Desktop/DL_train_CL_seagate.hdev')
        aug_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'augment_prepare'))
        preprocess_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'prepare_for_training'))
        # training_call = ha.HDevProcedureCall(ha.HDevProcedure.load_external('train_dl_model_PK'))
        # evaluation_call = ha.HDevProcedureCall(ha.HDevProcedure.load_external('Evaluation'))

        aug_call.set_input_control_param_by_name('AugmentationPercentage', int(AugmentationPercentage))
        aug_call.set_input_control_param_by_name('Rotation', int(Rotation))
        aug_call.set_input_control_param_by_name('Mirror', str(mirror))
        aug_call.set_input_control_param_by_name('BrightnessVariation', int(BrightnessVariation))
        aug_call.set_input_control_param_by_name('BrightnessVariationSpot', 0)
        aug_call.set_input_control_param_by_name('CropPercentage', 50)

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
        preprocess_call.set_input_control_param_by_name('ImageWidth', int(ImWidth))
        preprocess_call.set_input_control_param_by_name('ImageHeight', int(ImHeight))
        preprocess_call.set_input_control_param_by_name('ImageNumChannels', int(ImChannel))
        preprocess_call.set_input_control_param_by_name('ModelFileName',
                                                        ModelFileName)
        preprocess_call.set_input_control_param_by_name('BatchSize', int(BatchSize))
        preprocess_call.set_input_control_param_by_name('InitialLearningRate', float(InitialLearningRate))
        preprocess_call.set_input_control_param_by_name('Momentum', float(Momentum))
        preprocess_call.set_input_control_param_by_name('DLDeviceType', Runtime)
        preprocess_call.set_input_control_param_by_name('NumEpochs', int(NumEpochs))

        ChangeLearningRateEpochsList = ChangeLearningRateEpochs.split(',')
        ChangeLearningRateEpochs = [int(i) for i in ChangeLearningRateEpochsList]
        preprocess_call.set_input_control_param_by_name('ChangeLearningRateEpochs', ChangeLearningRateEpochs)

        lr_changeList = lr_change.split(',')
        lr_change = [float(i) for i in lr_changeList]
        preprocess_call.set_input_control_param_by_name('lr_change', lr_change)
        preprocess_call.set_input_control_param_by_name('WeightPrior', float(WeightPrior))
        preprocess_call.set_input_control_param_by_name('EvaluationIntervalEpochs', 1)
        class_penaltyList = class_penalty.split(',')
        Class_Penalty = [float(i) for i in class_penaltyList]
        preprocess_call.set_input_control_param_by_name('Class_Penalty', Class_Penalty)
        preprocess_call.set_input_control_param_by_name('GenParamName_augment', GenParamName_augment)
        preprocess_call.set_input_control_param_by_name('GenParamValue_augment', GenParamValue_augment)
        preprocess_call.execute()

        DLModelHandle = preprocess_call.get_output_control_param_by_name('DLModelHandle')
        DLDataset = preprocess_call.get_output_control_param_by_name('DLDataset')
        TrainParam = preprocess_call.get_output_control_param_by_name('TrainParam')

        return DLModelHandle, DLDataset, TrainParam


def training(DLModelHandle, DLDataset, TrainParam):
    program = ha.HDevProgram('C:/Users/930415/Desktop/DL_train_CL_seagate.hdev')
    training_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'train_dl_model_CE'))
    # proc_training = ha.HDevProcedure.load_external('train_dl_model_CE')
    # proc_call = ha.HDevProcedureCall(proc_training)

    training_call.set_input_control_param_by_name('DLModelHandle', DLModelHandle)
    training_call.set_input_control_param_by_name('DLDataset', DLDataset)
    training_call.set_input_control_param_by_name('TrainParam', TrainParam)
    training_call.set_input_control_param_by_name('Display_Ctrl', 1)
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
