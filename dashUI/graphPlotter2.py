import time
from halcon import *
import halcon as ha

counter = 0
iterationList =[]
lossList = []
epochList =[]
'''''
while counter == 1:

    TrainInfo = ha.read_dict('C:/Users/930415\Desktop/Chadle_Halcon_Scripts/TrainInfo.hdict', (), ())
    epoch_tuple = ha.get_dict_tuple(TrainInfo, 'epoch')
    loss_tuple = ha.get_dict_tuple(TrainInfo, 'mean_loss')
    num_iterations_per_epoch = ha.get_dict_tuple(TrainInfo, 'num_iterations_per_epoch')
    iteration = num_iterations_per_epoch[0] * epoch_tuple[0]

    if iteration not in iterationList:
        iterationList.append(iteration)
        epochList.append(epoch_tuple[0])
    if loss_tuple[0] not in lossList:
        lossList.append(loss_tuple[0])

    time.sleep(0.5)
#    if TrainInfo == ha.read_dict('C:/Users/930415\Desktop/Chadle_Halcon_Scripts/TrainInfo.hdict', (), ()):
 #       counter = 1

    print(iterationList)
    print(epochList)
    print(lossList)

Evaluation_Info = ha.read_dict('C:/Users/930415\Desktop/Chadle_Halcon_Scripts/EvaluationInfo.hdict', (), ())

TrainSet_result = ha.get_dict_tuple(Evaluation_Info, 'result_train')
TrainSet_result_global = ha.get_dict_tuple(TrainSet_result, 'global')
TrainSet_top1_error = ha.get_dict_tuple(TrainSet_result_global, 'top1_error')

ValidationSet_result = ha.get_dict_tuple(Evaluation_Info, 'result')
ValidationSet_result_global = ha.get_dict_tuple(ValidationSet_result, 'global')
ValidationSet_top1_error = ha.get_dict_tuple(ValidationSet_result_global, 'top1_error')

Epoch = ha.get_dict_tuple(Evaluation_Info, 'epoch')

TrainSet_top1_error_value = TrainSet_top1_error[0]
ValidationSet_top1_error_value = ValidationSet_top1_error[0]

print('TrainSet_top1_error: '+ str(TrainSet_top1_error[0]))
print('ValidationSet_top1_error: '+ str(ValidationSet_top1_error[0]))
print('Epoch: '+ str(Epoch[0]))
'''

#data = [HHandle{type: dl_model, id: 0x27f3e45ae20}, HHandle{type: dict, id: 0x27f3e2d4f40}, HHandle{type: dict, id: 0x27f3e3f6820}]

biglist = [377.0, 214.0, 50.0, 467.0]
x = ['cat', 'dog']

length = len(x)

z = [biglist[i:i + length] for i in range(0, len(biglist), length)]
print (z)