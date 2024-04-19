import numpy as np
import pandas as pd
import re
import os
import sys
import argparse
from natsort import natsorted
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

###################################################################   Parsing Command in WSL-Ubuntu ########################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--log_path', help='path to the  log_train files.')
parsed_args = parser.parse_args(sys.argv[1:])
log_path = parsed_args.log_path

#########################################################  Walking Through Folders of Training Sessions ###################################################################################
eval_avg_class_accuracy_list=list()
eval_accuracy_list=list()
hyperparam_arr=np.zeros((30,3),dtype=float) # store hyperparameter values for all training sessions
for folder_index,folder in enumerate(natsorted(os.listdir(log_path))):

    path_annotation = os.path.join(log_path, folder, "log_train.txt")
    file=open(str(path_annotation),'r')
    #print('=============================',folder,'================================')
    
    min_eval_mean_loss=None
    eval_avg_class_accuracy=0.0
    eval_accuracy=0
    flag=0

    file_str=file.read() # put each session's file in one big string
    #print(str)
    hyperparam_list=re.findall('batch_size=([0-9]+).*learning_rate=([0-9].[0-9]+).*momentum=([0-9].[0-9]+)',file_str) # regular expression technique to extract hyperparameters
    hyperparam_list=[list(x) for x in hyperparam_list] # convert list of a tuple into a list of a list 
    hyperparam=[float(x) for x in hyperparam_list[0]] # convert the list's elements inside the list, "hyperparam_list" into float numbers
    hyperparam_arr[folder_index,:]=hyperparam # put hyperparameters of each session in a separate row inside the array
    file.close()
    
    file=open(str(path_annotation),'r')
    for line in file:
       
       line=line.strip()
       
       if line.startswith('eval mean loss:') : 
          words=line.split(':')
          eval_mean_loss= words[-1]                                                        # take the last word in words list
          if (min_eval_mean_loss is None) or (min_eval_mean_loss>eval_mean_loss):
               min_eval_mean_loss = eval_mean_loss
               #print('eval mean loss: ',min_eval_mean_loss)
               flag=1
       elif line.startswith('eval accuracy:'):
          if flag ==1:
               words=line.split(':')
               eval_accuracy= words[-1]
               #print('eval accuracy: ',eval accuracy)

       elif line.startswith('eval avg class acc:'): 
          if flag ==1:
               words=line.split(':')
               eval_avg_class_accuracy= words[-1]
               #print('eval avg class acc: ',eval_avg_class_accuracy)
               flag=0   
       else:
            continue
       
    eval_avg_class_accuracy_list.append(float(eval_avg_class_accuracy)) 
    eval_accuracy_list.append((float(eval_accuracy),folder))

print(" optimum avg class accuracy in model: ", sorted(eval_avg_class_accuracy_list,reverse=True)[:5],'\n',"Optimum model accuracy: ",max(eval_avg_class_accuracy_list))
print('===================================================================================================================================================================================')
######################################################################## Indices of the elements before sorting ###########################################################################
indices=list()
for i in sorted(eval_avg_class_accuracy_list,reverse=True):
       indices.append(eval_avg_class_accuracy_list.index(i)) # Find Indices of the elements before sorting
#print(indices)
sessions=list()
for i in range(30):
   sessions.append('session'+str(i))
sessions_arr=np.array(sessions)
#print(hyperparam_arr[indices])
############################################################### Generate Excel Sheet ######################################################################################################
filepath = 'models.xlsx'
eval_avg_class_accuracy_arr=np.array(sorted(eval_avg_class_accuracy_list,reverse=True))
excel_mat=np.concatenate((hyperparam_arr[indices],eval_avg_class_accuracy_arr.reshape(30,1)),1) # horizontal concatenation
excel_mat=np.float64(excel_mat)
#print(excel_mat)
df = pd.DataFrame.from_records(excel_mat, index=sessions_arr[indices])
df.to_excel(filepath, header = ['Batch_size','Learning_rate','Momentum','validation average class accuracy'], index_label = 'Models')
   

