import os
import torch
import matplotlib.pyplot as plt
import pickle
import time
import logging

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from utils import *
from transforms import *
from dataset import *
from model import *


class Trainer:
    def __init__(self, data_dict):
        self.train_data_dir = data_dict['train_data_dir']
        print("train data:")
        print_tiff_filenames(self.train_data_dir)

        self.val_data_dir = data_dict['val_data_dir']
        print("validation data:")
        print_tiff_filenames(self.val_data_dir)

        self.project_dir = data_dict['project_dir']
        self.project_name = data_dict['project_name']

        self.disp_freq = data_dict['disp_freq']
        self.val_freq = data_dict['val_freq']
        self.train_continue = data_dict['train_continue']

        self.hyperparameters = data_dict['hyperparameters']

        self.model_name = self.hyperparameters['model_name']
        self.UNet_base = self.hyperparameters['UNet_base']
        self.stack_depth = self.hyperparameters['stack_depth']
        self.num_epoch = self.hyperparameters['num_epoch']
        self.batch_size = self.hyperparameters['batch_size']
        self.lr = self.hyperparameters['lr']
        self.patience = self.hyperparameters.get('patience', 10)

        self.device = get_device()

        self.results_dir, self.checkpoints_dir = create_result_dir(
            self.project_dir, self.project_name, self.hyperparameters, self.train_data_dir)
        self.train_results_dir, self.val_results_dir = create_train_val_dir(self.results_dir)

        self.writer = SummaryWriter(self.results_dir + '/tensorboard_logs')

    def save(self, checkpoints_dir, model, optimizer, epoch, best_val_loss):
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'hyperparameters': self.hyperparameters
        }, os.path.join(checkpoints_dir, 'best_model.pth'))

    def load(self, checkpoints_dir, model, device, optimizer):
        checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        dict_net = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(dict_net['model'])
        optimizer.load_state_dict(dict_net['optimizer'])
        epoch = dict_net['epoch']
        best_val_loss = dict_net.get('best_val_loss', float('inf'))
        self.hyperparameters = dict_net.get('hyperparameters', self.hyperparameters)

        print(f'Loaded {epoch}th network with hyperparameters: {self.hyperparameters}, best validation loss: {best_val_loss:.4f}')

        return model, optimizer, epoch, best_val_loss

    def get_model(self):
        if self.model_name == 'UNet3':
            return UNet3D_3(base=self.UNet_base).to(self.device)
        elif self.model_name == 'UNet4':
            return UNet3D_4(base=self.UNet_base).to(self.device)
        elif self.model_name == 'UNet5':
            return UNet3D_5(base=self.UNet_base).to(self.device)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def train(self):
        start_time = time.time()
        mean, std = compute_global_mean_and_std(self.train_data_dir, self.checkpoints_dir)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        transform_train = transforms.Compose([
            Normalize(mean, std),
            RandomCrop(output_size=(64,64)),
            RandomHorizontalFlip(),
            ToTensor()
        ])

        transform_inv_train = transforms.Compose([
            BackTo01Range(),
            ToNumpy()
        ])

        val_transform = transforms.Compose([
            Normalize(mean, std),
            CropToMultipleOf16Validation(),
            ToTensor(),
        ])

        dataset_train = VolumeSubstackDataset(self.train_data_dir, stack_depth=self.stack_depth, transform=transform_train)

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        dataset_val = ValidationDataset(self.val_data_dir, stack_depth=self.stack_depth, transform=val_transform)

        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        model = self.get_model()
        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), self.lr)

        st_epoch = 0
        best_val_loss = float('inf')
        patience_counter = 0

        if self.train_continue == 'on':
            model, optimizer, st_epoch, best_val_loss = self.load(self.checkpoints_dir, model, self.device, optimizer)
            model = model.to(self.device)

        for epoch in range(st_epoch + 1, self.num_epoch + 1):
            model.train()
            train_loss = 0.0

            for batch, data in enumerate(loader_train, 1):
                optimizer.zero_grad()
                input_stack, target_stack = [x.to(self.device) for x in data]
                if len(input_stack.shape) == 4:
                    input_stack = input_stack[np.newaxis, ...]
                    target_stack = target_stack[np.newaxis, ...]
                output_stack = model(input_stack)

                loss = criterion(output_stack, target_stack)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            if epoch % self.disp_freq == 0:
                input_stack = transform_inv_train(input_stack)
                target_stack = transform_inv_train(target_stack)
                output_stack = transform_inv_train(output_stack)

                for j in range(target_stack.shape[0]):
                    plt.imsave(os.path.join(self.train_results_dir, f"{j}_input.png"), input_stack[j, 0, :, :, 0], cmap='gray')
                    plt.imsave(os.path.join(self.train_results_dir, f"{j}_target.png"), target_stack[j, 0, :, :, 0], cmap='gray')
                    plt.imsave(os.path.join(self.train_results_dir, f"{j}_output.png"), output_stack[j, 0, :, :, 0], cmap='gray')

            avg_train_loss = train_loss / len(loader_train)
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)

            print(f'Epoch [{epoch}/{self.num_epoch}], Train Loss: {avg_train_loss:.4f}')

            avg_val_loss = float('inf')

            if epoch % self.val_freq == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_batch, val_data in enumerate(val_loader, 0):
                        val_input_stack, val_target_stack = [x.to(self.device) for x in val_data]
                        if len(input_stack.shape) == 4:
                            val_input_stack = val_input_stack[np.newaxis, ...]
                            val_target_stack = val_target_stack[np.newaxis, ...]
                        val_output_stack = model(val_input_stack)
                        val_loss += criterion(val_output_stack, val_target_stack)

                avg_val_loss = val_loss / len(val_loader)
                self.writer.add_scalar('Loss/val', avg_val_loss, epoch)

                print(f'Epoch [{epoch}/{self.num_epoch}], Validation Loss: {avg_val_loss:.10f}')

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save(self.checkpoints_dir, model, optimizer, epoch, best_val_loss)
                    patience_counter = 0
                    print(f"Saved best model at epoch {epoch} with validation loss {best_val_loss:.4f}.")
                else:
                    patience_counter += 1
                    print(f'Patience Counter: {patience_counter}/{self.patience}')

                if patience_counter >= self.patience:
                    print(f'Early stopping triggered after {epoch} epochs')
                    break

        self.writer.close()
