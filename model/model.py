import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, dropout_value=0.1):
        """This function instantiates all the model layers and assigns them as class variables.
        It has a single parameter dropout_value which is used to set the dropout value for all the layers.
        It uses the nn.Sequential() function to instantiate the layers. It has the following layers:
        1. Convolution layer 1: Input: 28x28x1, Output: 26x26x16, RF: 3x3
        2. Convolution layer 2: Input: 26x26x16, Output: 24x24x32, RF: 5x5
        3. Convolution layer 3: Input: 24x24x32, Output: 24x24x10, RF: 5x5
        4. MaxPooling layer 1: Input: 24x24x10, Output: 12x12x10, RF: 10x10
        5. Convolution layer 4: Input: 12x12x10, Output: 10x10x16, RF: 12x12
        6. Convolution layer 5: Input: 10x10x16, Output: 8x8x16, RF: 14x14
        7. Convolution layer 6: Input: 8x8x16, Output: 6x6x16, RF: 16x16
        8. Convolution layer 7: Input: 6x6x16, Output: 6x6x16, RF: 16x16
        9. GAP layer: Input: 6x6x16, Output: 1x1x16, RF: 28x28
        10. Convolution layer 8: Input: 1x1x16, Output: 1x1x10, RF: 28x28"""

        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
        )  # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )  # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
        )  # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
        )  # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
        )  # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
        )  # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value),
        )  # output_size = 6

        # OUTPUT BLOCK
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=6))  # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=10,
                kernel_size=(1, 1),
                padding=0,
                bias=False,
            ),
        )

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This function defines the forward pass of the model. It takes in a tensor x as input
        and passes it through each layer of the model, finally returning the output. it returns softmax of the output.
        """
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
