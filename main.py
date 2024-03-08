# Imports

## Add the modules to the system path
import os
import sys
sys.path.append(os.path.join(".."))

## Libs
from random import shuffle
import glob
import tifffile
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

## Own modules
import utils
from train import Trainer3D as Trainer



def main():


    if os.getenv('RUNNING_ON_SERVER') == 'true':

        path_results = os.path.join('/g', 'prevedel', 'members', 'Rauscher', 'projects', '3D-N2N-single_volume-2')
        path_dataset = os.path.join('/g', 'prevedel', 'members', 'Rauscher', 'data', 'OCT-3D-data')

    else:

        path_results = os.path.join('Z:\\', 'members', 'Rauscher', 'projects', '3D-N2N-single_volume-2')
        path_dataset = os.path.join('Z:\\', 'members', 'Rauscher', 'data', 'OCT-3D-data')


    # Select whether 3D-N2V or 3D-N2V2 and normal Unet or WUnet should be used #
    use_N2V2_setup = False
    use_WUNet_setup = True
    #**************************************************************************#

    # Enter the store path for the results and raw file here #
    # path_results = os.path.join('/g', 'prevedel', 'members', 'Rauscher', 'projects', '3D-N2N-single_volume')
    # path_results = os.path.join('Z:\\', 'members', 'Rauscher', 'projects', '3D-N2N-single_volume')
    if use_N2V2_setup:
        if use_WUNet_setup:
            path_results = os.path.join(path_results, "results_3D-WUNet-N2N2")
        else:
            path_results = os.path.join(path_results, "results_3D-N2N2")
    else:
        if use_WUNet_setup:
            path_results = os.path.join(path_results, "results_3D-WUNet-N2N")
        else:
            path_results = os.path.join(path_results, "results_3D-N2N")

    # path_dataset = os.path.join('Z:\\', 'members', 'Rauscher', 'data', 'OCT-3D-data')
    #********************************************************#

    # Create all the other paths based on the results folder

    # Make a folder to store results
    res_folder = os.path.join(path_results, 'training_results')
    os.makedirs(res_folder, exist_ok=True)

    # Make a folder to store the log files
    log_folder = os.path.join(path_results, 'log_files')
    os.makedirs(log_folder, exist_ok=True)

    log_train_folder = os.path.join(log_folder, 'train')
    os.makedirs(log_train_folder, exist_ok=True)

    log_val_folder = os.path.join(log_folder, 'val')
    os.makedirs(log_val_folder, exist_ok=True)

    # Make a folder for the checkpoints
    checkpoint_folder = os.path.join(path_results, 'checkpoints')
    os.makedirs(checkpoint_folder, exist_ok=True)

    # List all folders in the results folder to make sure all folder exists
    output_files = os.listdir(path_results)
    print("*****Output Folder*****")
    print("List of all folder in the results path:")
    print(output_files)
    print("***********************")

    # Select the path to data folder and also the specific file names if wanted #
    path_dataset_train_input = os.path.join(path_dataset, "input_train_dataset")
    path_dataset_train_target = os.path.join(path_dataset, "target_train_dataset")
    path_dataset_val_input = os.path.join(path_dataset, "input_validation_dataset")
    path_dataset_val_target = os.path.join(path_dataset, "target_validation_dataset")

    filenames_train_input = None 
    filenames_train_target = None 
    filenames_val_input = None 
    filenames_val_target = None 

    data_type_ending = ".tif"
    #***************************************************************************#

    ## Load image stack as dataset 

    if filenames_train_input is None:
        filenames_train_input = glob.glob(os.path.join(path_dataset_train_input, "*"+data_type_ending))
        filenames_train_target = glob.glob(os.path.join(path_dataset_train_target, "*"+data_type_ending))
    else:
        filenames_train_input = [os.path.join(path_dataset_train_input, file) for file in filenames_train_input] if type(filenames_train_input)==list else [filenames_train_input]
        filenames_train_target = [os.path.join(path_dataset_train_target, file) for file in filenames_train_target] if type(filenames_train_target)==list else [filenames_train_target]

    if filenames_val_input is None:
        filenames_val_input = glob.glob(os.path.join(path_dataset_val_input, "*"+data_type_ending))
        filenames_val_target = glob.glob(os.path.join(path_dataset_val_target, "*"+data_type_ending))
    else:
        filenames_val_input = [os.path.join(path_dataset_val_input, file) for file in filenames_val_input] if type(filenames_val_input)==list else [filenames_val_input]
        filenames_val_target = [os.path.join(path_dataset_val_target, file) for file in filenames_val_target] if type(filenames_val_target)==list else [filenames_val_target]

    print("On following file will be trained:")
    print("Input")
    print(*filenames_train_input, sep=",\n")
    print("Target")
    print(*filenames_train_target, sep=",\n")

    print("On following file will be validated:")
    print("Input")
    print(*filenames_val_input, sep=",\n")
    print("Target")
    print(*filenames_val_target, sep=",\n")

    files_train_input = [tifffile.imread(file) for file in filenames_train_input]
    files_train_target = None # [tifffile.imread(file) for file in filenames_train_target]
    files_val_input = [tifffile.imread(file) for file in filenames_val_input]
    files_val_target = [tifffile.imread(file) for file in filenames_val_target]

    # Select the training parameters #
    # T x Y x X
    input_size = [32, 64, 64]

    # #Training-to-#Validation ratio
    train_val_fraction = 0.5

    # Training epochs
    epoch = 300

    # Batch size
    batch_size = 16

    # Logger frequencies
    display_freq = 500
    model_storing_freq = 100
    #*********************************#

    # Parameter dictionary
    parameter_dict= {}
    # paths
    # In case norm-factors are stored somewhere, not necessary
    parameter_dict['dir_norm_factors'] = os.path.join("no_norm_factors_stored")
    parameter_dict['dir_checkpoint'] = checkpoint_folder
    parameter_dict['dir_log'] = log_folder
    parameter_dict['dir_result'] = res_folder
    # training state
    parameter_dict['train_continue'] = 'on'
    # hyperparameters
    parameter_dict['num_epoch'] = epoch
    # batch size
    parameter_dict['batch_size'] = batch_size
    # adam optimizer
    parameter_dict['lr'] = 0.001
    parameter_dict['optim'] = 'adam'
    parameter_dict['beta1'] = 0.5
    parameter_dict['beta2'] = 0.999
    # colormap
    parameter_dict['cmap'] = 'gray'
    # size of the input patches
    parameter_dict['ny'] = input_size[2]
    parameter_dict['nx'] = input_size[1]
    parameter_dict['nz'] = input_size[0]
    # channel dimension
    parameter_dict['nch'] = 1

    # Use the N2V2 setup
    parameter_dict['N2V2'] = use_N2V2_setup
    # Use the WUNet setup
    parameter_dict['WUNet'] = use_WUNet_setup
    # logger parameter
    parameter_dict['num_freq_disp'] = display_freq
    parameter_dict['num_freq_save'] = model_storing_freq
    # datasets
    parameter_dict['train_dataset'] = files_train_input
    parameter_dict['train_dataset_target'] = files_train_target
    minimum_train_stack_size = min([file.shape[0] for file in files_train_input])
    parameter_dict['val_dataset'] = [file[:-int(train_val_fraction*minimum_train_stack_size)] if minimum_train_stack_size>file.shape[0] else file for file in files_val_input]
    parameter_dict['val_dataset_target'] = [file[:-int(train_val_fraction*minimum_train_stack_size)] if minimum_train_stack_size>file.shape[0] else file for file in files_val_target]

    # Show the parameters
    print("***** Parameters *****")
    print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in parameter_dict.items()) + "}")
    print("**********************")

    # Generate Trainer
    trainer = Trainer(parameter_dict)
    # Start training
    print("*****Start of Training*****")
    trainer.train()
    print("*****End of Training*******")

if __name__ == '__main__':
    main()