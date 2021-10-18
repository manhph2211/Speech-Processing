import torch
import logging
import torch.nn as nn
from model_utils import *
logger = logging.getLogger(__name__)


class Xvector(nn.Module):
    def __init__(
        self,
        device="gpu",
        activation=nn.LeakyReLU,
        tdnn_blocks=5,
        tdnn_channels=[512, 512, 512, 512, 1500],
        tdnn_kernel_sizes=[5, 3, 3, 1, 1],
        tdnn_dilations=[1, 2, 3, 1, 1],
        lin_neurons=512,
        in_channels=40,
    ):

        super().__init__()
        self.blocks = nn.ModuleList()
        for block_index in range(tdnn_blocks):
            out_channels = tdnn_channels[block_index]
            self.blocks.extend(
                [
                    Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=tdnn_kernel_sizes[block_index],
                        dilation=tdnn_dilations[block_index],
                    ),
                    activation(),
                    BatchNorm1d(input_size=out_channels),
                ]
            )
            in_channels = tdnn_channels[block_index]
        self.blocks.append(StatisticsPooling())

        # Final linear transformation.
        self.blocks.append(
            Linear(
                input_size=out_channels * 2,  # mean + std,
                n_neurons=lin_neurons,
                bias=True,
                combine_dims=False,
            )
        )

    def forward(self, x, lens=None):

        for layer in self.blocks:
            try:
                x = layer(x, lengths=lens)
            except TypeError:
                x = layer(x)
        return x


class Classifier(Sequential):

    def __init__(
        self,
        input_shape,
        activation=nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=512,
        out_neurons=512,
    ):
        super().__init__(input_shape=input_shape)

        self.append(activation(), layer_name="act")
        self.append(BatchNorm1d, layer_name="norm")

        if lin_blocks > 0:
            self.append(Sequential, layer_name="DNN")

        # Adding fully-connected layers
        for block_index in range(lin_blocks):
            block_name = f"block_{block_index}"
            self.DNN.append(
                Sequential, layer_name=block_name
            )
            self.DNN[block_name].append(
                Linear,
                n_neurons=lin_neurons,
                bias=True,
                layer_name="linear",
            )
            self.DNN[block_name].append(activation(), layer_name="act")
            self.DNN[block_name].append(
                BatchNorm1d, layer_name="norm"
            )

        # Final Softmax classifier
        self.append(
           Linear, n_neurons=out_neurons, layer_name="out"
        )
        self.append(
            Softmax(apply_log=True), layer_name="softmax"
        )


if __name__ == '__main__':
    inputs = torch.rand(10, 50, 40)
    lin_t = Linear(input_shape=(10, 50, 40), n_neurons=100)
    output = lin_t(inputs)
    print(output.shape)
    # torch.Size([10, 50, 100])