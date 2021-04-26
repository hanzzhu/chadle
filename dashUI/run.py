import os
import halcon as ha

# import thread


running = True  # Global flag
# Chadle_Projects : Folder storing every thing
# Chadle_Data : Folder storing all projects
# Chadle_Halcon_Scripts : Folder storing Halcon Scripts, traininfo.hdict and evaluation.hdict
# Halcon_DL_library_files : Folder storing all projects

# Manually change Chadle_ProjectsDir if needed
Chadle_ProjectsDir = 'C:/Users/930415/Desktop/Chadle_Projects'

Chadle_DataDir = Chadle_ProjectsDir + '/Chadle_Data'
Chadle_Halcon_ScriptsDir = Chadle_ProjectsDir + '/Chadle_Halcon_Scripts'
Halcon_DL_library_filesDir = Chadle_ProjectsDir + '/Halcon_DL_library_files'

# Hdict files for plotting graph
TrainInfoDir = Chadle_Halcon_ScriptsDir + '/TrainInfo.hdict'
EvaluationInfoDir = Chadle_Halcon_ScriptsDir + '/EvaluationInfo.hdict'

# List of projects by getting folder names under Chadle_Data
ProjectList = next(os.walk(Chadle_DataDir))[1]


def setup_hdev_engine():
    """Setup HDevEngine by setting procedure search paths."""

    engine = ha.HDevEngine()
    engine.set_procedure_path('C:/Program Files/MVTec/HALCON-20.11-Progress/procedures')

    # path where dl_training_PK.hdl and dl_visualization_PK.hdl files are located
    engine.set_procedure_path(Halcon_DL_library_filesDir)
    program = ha.HDevProgram(Chadle_Halcon_ScriptsDir + '/DL_train_CL_seagate.hdev')
    aug_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'augment_prepare'))
    preprocess_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'prepare_for_training'))
    evaluation_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'Evaluation'))
    training_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'train_dl_model_CE'))
    # engine.set_procedure_path('E:/Customer evaluation/Seagate/HDev Engine_Python/dl_training_PK.hdpl')
    # engine.set_procedure_path('E:/Customer evaluation/Seagate/HDev Engine_Python/dl_visualization_PK.hdpl')
    return aug_call, preprocess_call, training_call, evaluation_call


def pre_process(ProjectName, Runtime, PretrainedModel, ImWidth, ImHeight, ImChannel,
                BatchSize, InitialLearningRate, Momentum, NumEpochs, ChangeLearningRateEpochs, lr_change, WeightPrior,
                class_penalty, AugmentationPercentage, Rotation, mirror, BrightnessVariation, BrightnessVariationSpot,
                RotationRange, ):
    call_list = setup_hdev_engine()
    aug_call = call_list[0]
    preprocess_call = call_list[1]
    training_call = call_list[2]
    evaluation_call = call_list[3]

    FileHandle = ha.open_file('mutex.dat', 'output')
    ha.fwrite_string(FileHandle, 0)
    if os.path.exists(TrainInfoDir):
        os.remove(TrainInfoDir)
    if os.path.exists(EvaluationInfoDir):
        os.remove(EvaluationInfoDir)
    # Search user input and match with project names
    # Upper case directory will be handled in Halcon, no need change back
    var = list((x for x in list(map(str.upper, ProjectList)) if ProjectName.upper() in x))
    print(var)
    if var:
        ProjectDir = Chadle_DataDir + '/' + var[0]

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

        # training_call = ha.HDevProcedureCall(ha.HDevProcedure.load_external('train_dl_model_PK'))
        # evaluation_call = ha.HDevProcedureCall(ha.HDevProcedure.load_external('Evaluation'))

        aug_call.set_input_control_param_by_name('AugmentationPercentage', int(AugmentationPercentage))
        aug_call.set_input_control_param_by_name('Rotation', int(Rotation))
        aug_call.set_input_control_param_by_name('Mirror', str(mirror))
        aug_call.set_input_control_param_by_name('BrightnessVariation', int(BrightnessVariation))
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
        preprocess_call.set_input_control_param_by_name('ImageWidth', int(ImWidth))
        preprocess_call.set_input_control_param_by_name('ImageHeight', int(ImHeight))
        preprocess_call.set_input_control_param_by_name('ImageNumChannels', int(ImChannel))
        preprocess_call.set_input_control_param_by_name('ModelFileName',
                                                        ModelFileName)
        preprocess_call.set_input_control_param_by_name('BatchSize', int(BatchSize))
        preprocess_call.set_input_control_param_by_name('InitialLearningRate', float(InitialLearningRate))
        preprocess_call.set_input_control_param_by_name('Momentum', float(Momentum))
        preprocess_call.set_input_control_param_by_name('DLDeviceType', 'cpu')
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
    call_list = setup_hdev_engine()
    aug_call = call_list[0]
    preprocess_call = call_list[1]
    training_call = call_list[2]
    evaluation_call = call_list[3]
    # proc_training = ha.HDevProcedure.load_external('train_dl_model_CE')
    # proc_call = ha.HDevProcedureCall(proc_training)

    training_call.set_input_control_param_by_name('DLModelHandle', DLModelHandle)
    training_call.set_input_control_param_by_name('DLDataset', DLDataset)
    training_call.set_input_control_param_by_name('TrainParam', TrainParam)
    training_call.set_input_control_param_by_name('Display_Ctrl', 0)
    training_call.set_input_control_param_by_name('StartEpoch', 0)

    training_call.execute()


def evaluation(ProjectName, Runtime, PretrainedModel, ImWidth, ImHeight, ImChannel,
               BatchSize, InitialLearningRate, Momentum, NumEpochs, ChangeLearningRateEpochs, lr_change, WeightPrior,
               class_penalty, AugmentationPercentage, Rotation, mirror, BrightnessVariation, BrightnessVariationSpot,

               ):
    call_list = setup_hdev_engine()
    aug_call = call_list[0]
    preprocess_call = call_list[1]
    training_call = call_list[2]
    evaluation_call = call_list[3]

    if ProjectName in ProjectList:
        ProjectDir = Chadle_DataDir + '/' + ProjectName

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

    EvalBatchSize = 1

    ImWidth = int(ImWidth)
    ImHeight = int(ImHeight)
    evaluation_call.set_input_control_param_by_name('BatchSize', EvalBatchSize)
    evaluation_call.set_input_control_param_by_name('ModelDir', ModelDir)
    evaluation_call.set_input_control_param_by_name('ExampleDataDir', SplitDir)
    evaluation_call.set_input_control_param_by_name('ImageWidth', ImWidth)
    evaluation_call.set_input_control_param_by_name('ImageHeight', ImHeight)
    evaluation_call.execute()

    output_EvalResults = evaluation_call.get_output_control_param_by_name('EvaluationResult')
    confusion_matrix_tuple = ha.get_dict_tuple(output_EvalResults, 'absolute_confusion_matrix')
    confusion_matrix_List = ha.get_full_matrix(confusion_matrix_tuple)
    values_inside_global = ha.get_dict_tuple(output_EvalResults, 'global')
    mean_precision = ha.get_dict_tuple(values_inside_global, 'mean_precision')
    mean_recall = ha.get_dict_tuple(values_inside_global, 'mean_recall')
    mean_f_score = ha.get_dict_tuple(values_inside_global, 'mean_f_score')

    listout = [confusion_matrix_List, mean_precision, mean_recall, mean_f_score]

    return listout


def get_TrainInfo():
    if os.path.isfile(TrainInfoDir):
        try:
            TrainInfo = ha.read_dict(TrainInfoDir, (), ())
            time_elapsed = ha.get_dict_tuple(TrainInfo, 'time_elapsed')
            time_elapsed = time_elapsed[0]
            time_remaining = ha.get_dict_tuple(TrainInfo, 'time_remaining')
            time_remaining = time_remaining[0]
            epoch_traininfo = ha.get_dict_tuple(TrainInfo, 'epoch')
            epoch_traininfo = epoch_traininfo[0]
            loss_tuple = ha.get_dict_tuple(TrainInfo, 'mean_loss')
            loss_tuple = loss_tuple[0]
            num_iterations_per_epoch = ha.get_dict_tuple(TrainInfo, 'num_iterations_per_epoch')
            num_iterations_per_epoch = num_iterations_per_epoch[0]
            iteration = num_iterations_per_epoch * epoch_traininfo

        except:
            TrainInfo = ha.read_dict(TrainInfoDir, (), ())
            time_elapsed = ha.get_dict_tuple(TrainInfo, 'time_elapsed')
            time_elapsed = time_elapsed[0]
            time_remaining = ha.get_dict_tuple(TrainInfo, 'time_remaining')
            time_remaining = time_remaining[0]
            epoch_traininfo = ha.get_dict_tuple(TrainInfo, 'epoch')
            epoch_traininfo = epoch_traininfo[0]
            loss_tuple = ha.get_dict_tuple(TrainInfo, 'mean_loss')
            loss_tuple = loss_tuple[0]
            num_iterations_per_epoch = ha.get_dict_tuple(TrainInfo, 'num_iterations_per_epoch')
            num_iterations_per_epoch = num_iterations_per_epoch[0]
            iteration = num_iterations_per_epoch * epoch_traininfo
        return time_elapsed, time_remaining, epoch_traininfo, loss_tuple, iteration
    else:
        return False


def get_EvaluationInfo():
    if os.path.isfile(EvaluationInfoDir):
        try:
            Evaluation_Info = ha.read_dict(EvaluationInfoDir, (), ())

            epoch_evaluation = ha.get_dict_tuple(Evaluation_Info, 'epoch')
            epoch_evaluation_value = epoch_evaluation[0]
            TrainSet_result = ha.get_dict_tuple(Evaluation_Info, 'result_train')
            TrainSet_result_global = ha.get_dict_tuple(TrainSet_result, 'global')
            TrainSet_top1_error = ha.get_dict_tuple(TrainSet_result_global, 'top1_error')

            ValidationSet_result = ha.get_dict_tuple(Evaluation_Info, 'result')
            ValidationSet_result_global = ha.get_dict_tuple(ValidationSet_result, 'global')
            ValidationSet_top1_error = ha.get_dict_tuple(ValidationSet_result_global, 'top1_error')

            TrainSet_top1_error_value = TrainSet_top1_error[0]
            ValidationSet_top1_error_value = ValidationSet_top1_error[0]

        except:
            Evaluation_Info = ha.read_dict(EvaluationInfoDir, (), ())

            epoch_evaluation = ha.get_dict_tuple(Evaluation_Info, 'epoch')
            epoch_evaluation_value = epoch_evaluation[0]
            TrainSet_result = ha.get_dict_tuple(Evaluation_Info, 'result_train')
            TrainSet_result_global = ha.get_dict_tuple(TrainSet_result, 'global')
            TrainSet_top1_error = ha.get_dict_tuple(TrainSet_result_global, 'top1_error')

            ValidationSet_result = ha.get_dict_tuple(Evaluation_Info, 'result')
            ValidationSet_result_global = ha.get_dict_tuple(ValidationSet_result, 'global')
            ValidationSet_top1_error = ha.get_dict_tuple(ValidationSet_result_global, 'top1_error')

            TrainSet_top1_error_value = TrainSet_top1_error[0]
            ValidationSet_top1_error_value = ValidationSet_top1_error[0]
        return epoch_evaluation_value, TrainSet_top1_error_value, ValidationSet_top1_error_value
    else:
        return False


def getImageCategories(ProjectName):
    labels = []
    if ProjectName in ProjectList:
        ProjectDir = Chadle_DataDir + '/' + ProjectName
        HalconImageDir = ProjectDir + '/Image'

        categoriesDir = HalconImageDir + '/Train'
        categoriesRaw = [y[0] for y in os.walk(categoriesDir)]
        categoriesRaw.remove(categoriesRaw[0])
        categories = []
        for i in range(len(categoriesRaw)):
            categories.append(os.path.basename(categoriesRaw[i]))

        for j in range(len(categories)):
            Prediction = 'Prediction:' + categories[j] + '\n'
            for k in range(len(categories)):
                labelString = Prediction + 'Truth: ' + categories[k]
                labels.append(labelString)

    return categories, labels
