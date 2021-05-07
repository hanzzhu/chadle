import halcon as ha

Chadle_ProjectsDir = 'C:/Users/930415/Desktop/Chadle_Projects'

Chadle_DataDir = Chadle_ProjectsDir + '/Chadle_Data'
Chadle_Halcon_ScriptsDir = Chadle_ProjectsDir + '/Chadle_Halcon_Scripts'
Halcon_DL_library_filesDir = Chadle_ProjectsDir + '/Halcon_DL_library_files'

def setup_hdev_engine():
    """Setup HDevEngine by setting procedure search paths."""

    engine = ha.HDevEngine()
    engine.set_procedure_path('C:/Program Files/MVTec/HALCON-20.11-Progress/procedures')

    # path where dl_training_PK.hdl and dl_visualization_PK.hdl files are located
    engine.set_procedure_path(Halcon_DL_library_filesDir)
    program = ha.HDevProgram(Chadle_Halcon_ScriptsDir + '/DL_Train_OD_seagate.hdev')
    aug_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'augment_prepare'))
    preprocess_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'prepare_for_training'))
    evaluation_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'Evaluation'))
    training_call = ha.HDevProcedureCall(ha.HDevProcedure.load_local(program, 'train_dl_model_CE'))
    # engine.set_procedure_path('E:/Customer evaluation/Seagate/HDev Engine_Python/dl_training_PK.hdpl')
    # engine.set_procedure_path('E:/Customer evaluation/Seagate/HDev Engine_Python/dl_visualization_PK.hdpl')
    return aug_call, preprocess_call, training_call, evaluation_call