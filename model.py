import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_channels, signal_length, f1=8, ke=32, f2=16, num_classes=4, dropout_prob=0.5):
        super(CNNModel, self).__init__()
        
        # Step 1: First convolutional layer
        self.conv1 = nn.Conv2d(1, f1, (1, ke), padding=(0, ke // 2))
        self.bn1 = nn.BatchNorm2d(f1)

        # Step 2: Channel-wise convolution
        # Using 2 [22, 1] kernels per group in f1 groups (corresponding to 2*f1)
        # This value must be a natural number multiple of f1
        self.conv2 = nn.Conv2d(f1, 2 * f1, (num_channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(2 * f1)
        self.elu2 = nn.ELU()

        # Step 3: Average pooling and Dropout
        self.pool2 = nn.AvgPool2d((1, 8), ceil_mode=True)
        self.dropout2 = nn.Dropout(dropout_prob)

        # Step 4: SeparableConv2D
        self.sep_conv = nn.Conv2d(2 * f1, f2, (1, 16), padding=(0, 16 // 2), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.elu3 = nn.ELU()

        # Step 5: Average pooling and Dropout
        self.pool3 = nn.AvgPool2d((1, 8), ceil_mode=True)
        self.dropout3 = nn.Dropout(dropout_prob)

        # Step 6: Calculate the input size for the fully connected layer
        self.flattened_size = f2 * 16  # Test shows size is 256 when f2=16

        # Step 7: Fully connected layer
        self.fc = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):  # x: [batchsize, num_channels, time_point]
        x = x.unsqueeze(1)  # [B, 1, 22, 1000]

        # Step 1: First convolutional layer [B, 8, 22, 1000]
        x = self.conv1(x)
        x = self.bn1(x)

        # Step 2: Channel-wise convolution [B, 16, 1, 1000]
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)

        # Step 3: Average pooling and Dropout [B, 16, 1, 125]
        x = self.pool2(x)
        x = self.dropout2(x)

        # Step 4: SeparableConv2D [B, 16, 1, 125]
        x = self.sep_conv(x)
        x = self.bn3(x)
        x = self.elu3(x)

        # Step 5: Average pooling and Dropout [B, 16, 1, 16]
        x = self.pool3(x)
        x = self.dropout3(x)

        # Step 6: Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x