from typing import Tuple, List, Union
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torch import Tensor
from options import parse_opts
from UNet import UNet, UNetConfig
import torch.optim as optim
from consistency_generator import ImprovedConsistencyTraining, pseudo_huber_loss
from inference import ConsistencySamplingAndEditing
from torchvision.utils import make_grid
from visdom import Visdom
import torchvision.utils as vutils
from torchinfo import summary
import numpy as np
import random
import torch
import os
import glob
# Decide which device we want to run on
ngpu = 1
device = torch.device("cuda:1" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
args = parse_opts()
viz = Visdom(env=args.env)

# Set seed for reproducibility
seed_value = 42  
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data module configuration
@dataclass
class ImageDataModuleConfig:
    data_dir: str
    image_size: Tuple[int, int]
    batch_size: int
    num_workers: int
    pin_memory: bool = True
    persistent_workers: bool = True

# Data module class
class ImageDataModule:
    def __init__(self, config: ImageDataModuleConfig) -> None:
        self.config = config
        self.dataset = None

    def setup(self) -> None:
        transform = T.Compose([
            T.Resize(self.config.image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1),
        ])
        self.dataset = ImageFolder(self.config.data_dir, transform=transform)

    def get_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

# Usage example
config = ImageDataModuleConfig(
    data_dir=args.data_dir,
    image_size=args.image_size,
    batch_size=args.batch_size,
    num_workers=args.num_workers
)
data_module = ImageDataModule(config)
data_module.setup()
train_loader = data_module.get_dataloader(shuffle=True)
print(f"train_loader len is {len(train_loader)}")

# Extract the dataset name from the data_dir
dataset_name = os.path.basename(os.path.normpath(args.data_dir))
checkpoint_dir = os.path.join("checkpoints", dataset_name)

# Create the dataset-specific checkpoint folder if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

# Check if train_continue option is provided
if args.train_continue:
    # Get the list of model paths
    model_paths = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*"))

    if model_paths:
        # Get the latest model path
        latest_model_path = max(model_paths, key=os.path.getctime)
        print(f"Model loaded from {latest_model_path}")

        # Load the latest model
        model = UNet.from_pretrained(latest_model_path)
    else:
        print(f"No existing models found in {checkpoint_dir}. Train from scratch.")
        # If not continuing, create a new model
        model = UNet(UNetConfig()).to(device)

else:
    print(f"No latest models found for {dataset_name}. Train from scratch.")
    # If not continuing, create a new model
    model = UNet(UNetConfig()).to(device)


# Model configuration and summary
summary(model, input_size=((1, 3, 32, 32), (1,)))

# Optimizer and scheduler setup
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.995))
scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-5, total_iters=1000
)

# Training configuration
@dataclass
class TrainingConfig:
    lr: float = args.lr
    betas: Tuple[float, float] = (0.9, 0.995)
    lr_scheduler_start_factor = args.lr_scheduler_start_factor
    lr_scheduler_iters = args.lr_scheduler_iters
    sample_every_n_steps = args.sample_every_n_steps
    num_samples = args.num_samples
    sampling_sigmas: Tuple[Tuple[int, ...], ...] = (
        (80,),
        (80.0, 0.661),
        (80.0, 24.4, 5.84, 0.9, 0.661),
    )

# Training configuration instance
training_config = TrainingConfig()

# Function for sampling and logging samples
@torch.no_grad()
def __sample_and_log_samples(batch: Union[Tensor, List[Tensor]], config: TrainingConfig, global_step) -> None:
    if isinstance(batch, list):
        batch = batch[0]

    # Ensure the number of samples does not exceed the batch size
    num_samples = min(config.num_samples, batch.shape[0])

    # Log ground truth samples
    __log_images(
        batch[:num_samples].detach(),
        "ground_truth",
        global_step,
        "ground_truth_window",
        window_size=(200, 200),
    )

    inpainting = ConsistencySamplingAndEditing()
    for sigmas in config.sampling_sigmas:
        samples = inpainting(
            model,
            batch.to(device),
            sigmas,
            clip_denoised=True,
            verbose=True,
        )
        samples = samples.clamp(min=-1.0, max=1.0)

        __log_images(
            samples,
            f"generated_samples-sigmas={sigmas}",
            global_step,
            f"generated_samples_window_{sigmas}",
            window_size=(400, 400),
        )

# Function for logging images
@torch.no_grad()
def __log_images(
    images: Tensor,
    title: str,
    global_step: int,
    window_name: str,
    window_size: Tuple[int, int],
) -> None:
    images = images.detach().float()

    grid = make_grid(
        images.clamp(-1.0, 1.0), value_range=(-1.0, 1.0), normalize=True, nrow=8
    )

    grid = grid.cpu().numpy()
    if grid.min() < 0:
        grid = (grid + 1) / 2
    width, height = window_size
    viz.images(
        grid,
        nrow=4,
        opts=dict(
            title=f"{title} at step {global_step}",
            caption=f"{title}",
            width=width,
            height=height,
        ),
        win=window_name,
    )

# Function for training one epoch
def train_one_epoch(epoch_index, epoch_length, device, config: TrainingConfig):
    running_loss = 0.
    last_loss = 0.
    global global_step
    for batch_idx, batch in enumerate(train_loader):
        #print(f"batch_idx is {batch_idx}")
        if isinstance(batch, list):
            batch = batch[0]
        viz.images( vutils.make_grid( batch, normalize=True, nrow=8 ), win="consistency_batch", 
                   opts=dict( title="train_batch image", caption="train_batch image",width=300, height=300,) )

        optimizer.zero_grad()

        consistency_training = ImprovedConsistencyTraining()

        global_step = epoch_index * epoch_length + batch_idx
        max_steps = EPOCHS * epoch_length
        output = consistency_training(model, batch.to(device), global_step, max_steps)
        loss = (pseudo_huber_loss(output.predicted, output.target) * output.loss_weights).mean()
        loss.backward()
        global_step += 1
        optimizer.step()
        running_loss += loss.item()

        viz.line(
            [loss.detach().cpu().numpy()],
            [global_step],
            win="consis_loss_iteration",
            opts=dict(title="loss over iteration"),
            update="append"
        )

        if batch_idx % len(train_loader) == 0:
            last_loss = running_loss / len(train_loader)
            running_loss = 0.

    return last_loss

# Main training loop

EPOCHS = int( args.max_steps / len(train_loader) )
global_step = 0  
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    

    avg_loss = train_one_epoch(epoch, len(train_loader) ,device, training_config)

    
   # print(f'trainning loss {avg_loss} at epoch {epoch+1}')
    
    
    viz.line( [avg_loss], [ epoch ],win="consis_loss_epoch", opts=dict(title="loss over epoch time"), update="append")
    if epoch % args.sample_every_n_steps == 0 or epoch == 0 or epoch == EPOCHS:
        model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}")
        model.save_pretrained(model_path)

        batch, _ = next(iter(train_loader))
        __sample_and_log_samples(batch.to(device), training_config, global_step)
    scheduler.step()


#    # Specify the path where the model is saved
# model_path = "checkpoints/"
#
# # Load the model
# loaded_model = UNet.from_pretrained(model_path)
#
# ###inpainting
# batch, _ = next(iter(train_loader))
# # Now you can use `loaded_model` for inference or further training
# # Create an instance of the class, optionally specify sigma_min and sigma_data
# inpainting = ConsistencySamplingAndEditing()
#
# # Now call the instance with the appropriate arguments for the __call__ method
# result = inpainting(loaded_model, batch.cpu(), (80.0, 24.4, 5.84, 0.9, 0.661))
#
