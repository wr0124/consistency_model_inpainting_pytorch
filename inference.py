from typing import Any, Callable, Iterable, Optional, Union, Tuple
from dataclasses import asdict, dataclass
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils import pad_dims_like
from consistency_generator import model_forward_wrapper

from UNet import UNet
from torchvision import transforms as T

from visdom import Visdom
import torchvision.utils as vutils
from options import parse_opts
args = parse_opts()
viz = Visdom(env=args.env)
#
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class ConsistencySamplingAndEditing:
    """Implements the Consistency Sampling and Zero-Shot Editing algorithms.

    Parameters
    ----------
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_data: float = 0.5) -> None:
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

    def __call__(
        self,
        model: nn.Module,
        y: Tensor,
        sigmas: Iterable[Union[Tensor, float]],
        mask: Optional[Tensor] = None,
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        start_from_y: bool = True,
        add_initial_noise: bool = True,
        clip_denoised: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """Runs the sampling/zero-shot editing loop.

        With the default parameters the function performs consistency sampling.

        Parameters
        ----------
        model : nn.Module
            Model to sample from.
        y : Tensor
            Reference sample e.g: a masked image or noise.
        sigmas : Iterable[Union[Tensor, float]]
            Decreasing standard deviations of the noise.
        mask : Tensor, default=None
            A mask of zeros and ones with ones indicating where to edit. By
            default the whole sample will be edited. This is useful for sampling.
        transform_fn : Callable[[Tensor], Tensor], default=lambda x: x
            An invertible linear transformation. Defaults to the identity function.
        inverse_transform_fn : Callable[[Tensor], Tensor], default=lambda x: x
            Inverse of the linear transformation. Defaults to the identity function.
        start_from_y : bool, default=False
            Whether to use y as an initial sample and add noise to it instead of starting
            from random gaussian noise. This is useful for tasks like style transfer.
        add_initial_noise : bool, default=True
            Whether to add noise at the start of the schedule. Useful for tasks like interpolation
            where noise will alerady be added in advance.
        clip_denoised : bool, default=False
            Whether to clip denoised values to [-1, 1] range.
        verbose : bool, default=False
            Whether to display the progress bar.
        **kwargs : Any
            Additional keyword arguments to be passed to the model.

        Returns
        -------
        Tensor
            Edited/sampled sample.
        """
        # Set mask to all ones which is useful for sampling and style transfer
        if mask is None:
            mask = torch.ones_like(y)

        # Use y as an initial sample which is useful for tasks like style transfer
        # and interpolation where we want to use content from the reference sample
        x = y if start_from_y else torch.zeros_like(y)

        #creat mask
        mask_fill_one = T.RandomErasing(p=1.0, scale=(0.2, 0.2), ratio=(0.5, 0.5), value =1 )
        zero_tensor = torch.zeros_like(x)
        mask = mask_fill_one(zero_tensor)
        # Sample at the end of the schedule
        y = self.__mask_transform(x, y, mask, transform_fn, inverse_transform_fn)
        # For tasks like interpolation where noise will already be added in advance we
        # can skip the noising process
        y = y * (1-mask)
        viz.images( vutils.make_grid( y, normalize=True, nrow=8 ), win="consistency_masked_input", 
                   opts=dict( title="inpainting_sigma1_input image", caption="inpainting_sigma1_input image",width=300, height=300,) )


        x = y + sigmas[0] * torch.randn_like(y) if add_initial_noise else y
        x = self.__mask_transform(x, y, mask, transform_fn, inverse_transform_fn)
        sigma = torch.full((x.shape[0],), sigmas[0], dtype=x.dtype, device=x.device)
        x = model_forward_wrapper(
            model, x, sigma, self.sigma_data, self.sigma_min, **kwargs
        )
        if clip_denoised:
            x = x.clamp(min=-1.0, max=1.0)
        x = self.__mask_transform(x, y, mask, transform_fn, inverse_transform_fn)

        viz.images(
            vutils.make_grid(
                x.to(dtype=torch.float32), normalize=True, nrow=int(x.shape[0] / 4)
            ),
            win="consistency_test3",
            opts=dict(
                title="x_T from sigma1 image",
                caption="x_T from sigma1 image",
                width=500,
                height=500,
            ),
        )
        # Progressively denoise the sample and skip the first step as it has already
        # been run
        pbar = tqdm(sigmas[1:], disable=(not verbose))
        for sigma in pbar:
            pbar.set_description(f"sampling (σ={sigma:.4f})")

            sigma = torch.full((x.shape[0],), sigma, dtype=x.dtype, device=x.device)
            x = x + pad_dims_like(
                (sigma**2 - self.sigma_min**2) ** 0.5, x
            ) * torch.randn_like(x)

            x = self.__mask_transform(x, y, mask, transform_fn, inverse_transform_fn)
            viz.images(
                vutils.make_grid(
                    x.to(dtype=torch.float32), normalize=True, nrow=int(x.shape[0] / 4)
                ),
                win="consistency_test4",
                opts=dict(
                    title="estimation x_t image",
                    caption="estimation x_t image",
                    width=500,
                    height=500,
                ),
            )


            x = model_forward_wrapper(
                model, x, sigma, self.sigma_data, self.sigma_min, **kwargs
            )

            viz.images(
                vutils.make_grid(
                    x.to(dtype=torch.float32), normalize=True, nrow=int(x.shape[0] / 4)
                ),
                win="consistency_test5",
                opts=dict(
                    title="function x_t image",
                    caption="function x_t image",
                    width=500,
                    height=500,
                ),
            )
            if clip_denoised:
                x = x.clamp(min=-1.0, max=1.0)
            x = self.__mask_transform(x, y, mask, transform_fn, inverse_transform_fn)

            viz.images(
                vutils.make_grid(
                    x.to(dtype=torch.float32), normalize=True, nrow=int(x.shape[0] / 4)
                ),
                win="consistency_test6",
                opts=dict(
                    title="final x_t image",
                    caption="final x_t image",
                    width=500,
                    height=500,
                ),
            )
        return x
    def __mask_transform(
        self,
        x: Tensor,
        y: Tensor,
        mask: Tensor,
        transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
        inverse_transform_fn: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> Tensor:
        return inverse_transform_fn(transform_fn(y) * (1.0 - mask) + x * mask)

#dataset
@dataclass
class ImageDataModuleConfig:
    data_dir: str
    image_size: Tuple[int, int]
    batch_size: int
    num_workers: int
    pin_memory: bool = True
    persistent_workers: bool = True

class ImageDataModule:
    def __init__(self, config: ImageDataModuleConfig) -> None:
        self.config = config
        self.dataset = None

    def setup(self) -> None:
        transform = T.Compose(
            [
                T.Resize(self.config.image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Lambda(lambda x: (x * 2) - 1),
            ]
        )
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
#
# #usage example
# config = ImageDataModuleConfig(
#     data_dir='/data3/juliew/datasets/butterflies/',
#     image_size=(32, 32),
#     batch_size=32,
#     num_workers=12
# )
# data_module = ImageDataModule(config)
# data_module.setup()
# train_loader = data_module.get_dataloader(shuffle=True)
#
#
# model_path = "/data3/juliew/projet2_diffusion/joligen/pytorch_consistency/checkpoints/"
#
# # Load the model
# loaded_model = UNet.from_pretrained(model_path)
# ###inpainting
# batch, _ = next(iter(train_loader))
#
# # Assuming the rest of your setup code is correct and above this snippet
#
# inpainting = ConsistencySamplingAndEditing()
#
# sigmas =(80.0, 24.4, 5.84, 0.9, 0.661)
#
#
# model = loaded_model.to(device)
# y = batch.to(device)  # Move your batch data to the correct device
#
# result = inpainting(model, y, sigmas)
#
# # For instance, visualize using Visdom
# viz.images(
#     vutils.make_grid(result.detach().cpu(), normalize=True, nrow=8),
#     win="consistency_result",
#     opts=dict(title="Consistency Sampling Result", width=300, height=300)
# )
#
