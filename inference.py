# Imports

## Add the modules to the system path
import os
import sys
sys.path.append(os.path.join(".."))

## Libs
import numpy as np
import glob
import tifffile
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

## Own modules
import utils
from transformations import PercentileNormalize3D, PercentileDenormalize3D, ZCrop3D, ToTensor3D, ToNumpy3D
from dataset import InferenceDataset3D
from network3D import Noise2NoiseUNet3D,WUNet3D

# Select whether 2D-T N2N setup was used or 3D-N2N #
used_2D_T_N2N = False
#**************************************************#

# Select whether 3D-N2V or 3D-N2V2 and normal Unet or WUnet should be used #
used_N2V2_setup = False
used_WUNet_setup = True 
#**************************************************#

# Enter the store path for the results and raw file here #
path_results = os.path.join('Z:\\', 'members', 'Rauscher', 'projects', '3D-N2N-single_volume')
if used_N2V2_setup:
    if used_2D_T_N2N:
        if used_WUNet_setup:
            path_results = os.path.join(path_results, "results_2D-T-WUNet-N2N2")
        else:
            path_results = os.path.join(path_results, "results_2D-T-N2N2")
    else:
        if used_WUNet_setup:
            path_results = os.path.join(path_results, "results_3D-WUNet-N2N2")
        else:
            path_results = os.path.join(path_results, "results_3D-N2N2")
else:
    if used_2D_T_N2N:
        if used_WUNet_setup:
            path_results = os.path.join(path_results, "results_2D-T-WUNet-N2N")
        else:
            path_results = os.path.join(path_results, "results_2D-T-N2N")
    else:
        if used_WUNet_setup:
            path_results = os.path.join(path_results, "results_3D-WUNet-N2N")
        else:
            path_results = os.path.join(path_results, "results_3D-N2N")

if used_2D_T_N2N:
    path_dataset = os.path.join("..", "data", "3PM-2DT-data")
else:
    path_dataset = os.path.join('Z:\\', 'members', 'Rauscher', 'data', 'OCT-3D-data')
#********************************************************#
    
# Create a folder for the inference based on the results folder

# Make a folder to store the inference
inference_folder = os.path.join(path_results, 'inference_results-whole_volume_')
os.makedirs(inference_folder, exist_ok=True)

# Define path to the checkpoint folder
checkpoint_folder = os.path.join(path_results, 'checkpoints')

## Load image stack for inference
filenames = glob.glob(os.path.join(path_dataset, "*", "*.tif"))
print("Following files will be denoised:")
print(*filenames, sep='\n')

info_file = tifffile.imread(filenames[0])

# Select the inference parameters #

# Crop top and bottom z-slices to prevent artefacts
z_crop_width = (2, 2)
# Determine which model should be used
use_lowest_loss_model = True

#********************************#

# datatype of the original data
data_type = info_file.dtype
print("The data type of the raw data is:   ", data_type)

# calculate the norm. factors
print("\nThe norm. factors are: ")
min_img, max_img = utils.calc_normfactors(info_file)

# check if GPU is accessable
if torch.cuda.is_available():
    print("\nGPU will be used.")
    device = torch.device("cuda:0")
else:
    print("\nCPU will be used.")
    device = torch.device("cpu")

## Data handling
# Use the right back-conversation
if data_type == np.uint16:
    norm_func = utils.NormFloat2UInt16(percent=1.0)
elif data_type == np.int16:
    norm_func = utils.NormFloat2Int16(percent=1.0)
else:
    norm_func = utils.NormFloat2UInt8(percent=1.0)

## Transformation
transform = transforms.Compose([PercentileNormalize3D(mi=min_img, ma=max_img),
                                ToTensor3D()
                                ])
transform_inv = transforms.Compose([ToNumpy3D(),
                                    ZCrop3D(z_crop_width),
                                    PercentileDenormalize3D(mi=min_img, ma=max_img),
                                    norm_func
                                   ])

## Load pretrained model
if used_WUNet_setup:
    net = WUNet3D(  in_channels = 1,
                    out_channels = 1,
                    is_N2V2_setup=used_N2V2_setup,
                    final_sigmoid = False).to(device)
else:
    net = Noise2NoiseUNet3D(in_channels = 1,
                            out_channels = 1,
                            is_N2V2_setup=used_N2V2_setup,
                            final_sigmoid = False).to(device)

net, st_epoch = utils.load(checkpoint_folder, net, device, use_best=use_lowest_loss_model)

for file in tqdm(filenames):
    
    # Load image
    
    # data
    image_stack_temp = tifffile.imread(file)
    ## Extend the image stack by the slice, which will be cropped
    image_stack_temp_pad = np.pad(image_stack_temp, ((z_crop_width[0], z_crop_width[1]), (0,0), (0,0)))
    image_stack_temp_pad = utils.norm_by_datatype(image_stack_temp_pad)
    ## Add batch and channel dimension
    image_stack_temp_pad = image_stack_temp_pad[None, ..., None]
    
    ## Applying the model

    ## generate the output array

    input_all = np.empty(image_stack_temp.shape, dtype=data_type)
    output_all = np.empty(image_stack_temp.shape, dtype=data_type)

    # Apply the model
    with torch.no_grad():
        net.eval()
        input = transform(image_stack_temp_pad)

        input = input.to(device)
        # forward net
        output = net(input)
        ## transform data back
        output_all = transform_inv(output).squeeze()
        input_all = transform_inv(input).squeeze()
        
    # Store the image stack
    file_store_name_out = os.path.join(inference_folder, file.split(os.sep)[-1][:-4] + '-best_model-denoised.tif' if use_lowest_loss_model else '-latest_model-denoised.tif')
    tifffile.imwrite(file_store_name_out, np.rint(output_all).astype(data_type))
    file_store_name_in = os.path.join(inference_folder, file.split(os.sep)[-1][:-4] + '-input.tif')
    tifffile.imwrite(file_store_name_in, np.rint(input_all).astype(data_type))

    plt.figure(figsize=(10,30))

file = tifffile.imread(filenames[-1])
ind = file.shape[0]//2

plt.subplot(131)
plt.title("Raw (Original)")
plt.imshow(file[ind], cmap="gray")

plt.subplot(132)
plt.title("Raw (Input Data)")
plt.imshow(input_all[ind], cmap="gray")

plt.subplot(133)
plt.title("Denoised")
plt.imshow(output_all[ind], cmap="gray")

plt.tight_layout()
plt.show()