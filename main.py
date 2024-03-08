# Imports

## Add the modules to the system path
import os
import sys
sys.path.append(os.path.join(".."))

## Libs
from random import shuffle
import glob
import logging
import tifffile
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

## Own modules
import utils
from train import Trainer3D as Trainer

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


logging.basicConfig(filename='logging.log',  # Log filename
                    filemode='a',  # Append mode, so logs are not overwritten
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp
                    level=logging.INFO,  # Logging level
                    datefmt='%Y-%m-%d %H:%M:%S')  # Timestamp formatlog_file = open('logfile.log', 'w', buffering=1)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set logging level for console
logging.getLogger('').addHandler(console_handler)

# Redirect stdout and stderr to logging
sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)


def main():


    if os.getenv('RUNNING_ON_SERVER') == 'true':

        path_results = os.path.join('/g', 'prevedel', 'members', 'Rauscher', 'projects', '3D-N2N-single_volume-3')
        path_dataset = os.path.join('/g', 'prevedel', 'members', 'Rauscher', 'data', 'big_data_small')

    else:

        path_results = os.path.join('Z:\\', 'members', 'Rauscher', 'projects', '3D-N2N-single_volume-3')
        # path_dataset = os.path.join('Z:\\', 'members', 'Rauscher', 'data', 'big_data_small')
        path_dataset = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'big_data_small')


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

    # Make a folder for the checkpoints
    checkpoint_folder = os.path.join(path_results, 'checkpoints')
    os.makedirs(checkpoint_folder, exist_ok=True)

    # List all folders in the results folder to make sure all folder exists
    output_files = os.listdir(path_results)
    print("*****Output Folder*****")
    print("List of all folder in the results path:")
    print(output_files)
    print("***********************")

    data_type_ending = ".TIFF"
    #***************************************************************************#

    ## Load image stack as dataset 
    files_train_input = []

    for subdir, _, files in os.walk(path_dataset):
        sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
        for f in sorted_files:
            print(f)
            full_path = os.path.join(subdir, f)
            files_train_input.append(tifffile.imread(full_path))


    files_train_target = None # [tifffile.imread(file) for file in filenames_train_target]

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
    model_storing_freq = 10
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
    parameter_dict['train_continue'] = 'off'
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