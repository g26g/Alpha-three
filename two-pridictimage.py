import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.decomposition import PCA
import scipy.io as sio
import numpy as np
import torch.optim as optim
from operator import truediv
from sklearn.metrics import cohen_kappa_score
import spectral

import os
import numpy as np
from PIL import Image



# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# 网络骨架
class HybridSN(nn.Module):
    def __init__(self, num_classes=9, self_attention=False):
        super(HybridSN, self).__init__()
        # out = (width - kernel_size + 2*padding)/stride + 1
        # => padding = ( stride * (out-1) + kernel_size - width)
        # 这里因为 stride == 1 所有卷积计算得到的padding都为 0

        # 默认不使用注意力机制
        self.self_attention = self_attention

        # 3D卷积块
        self.block_1_3D = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=(7, 3, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=8,
                out_channels=16,
                kernel_size=(5, 3, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True)
        )

        if self_attention:
            self.channel_attention_1 = ChannelAttention(256)
            self.spatial_attention_1 = SpatialAttention(kernel_size=7)

        # 2D卷积块
        self.block_2_2D = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=64,
                kernel_size=(3, 3)
            ),
            nn.ReLU(inplace=True)
        )

        if self_attention:
            self.channel_attention_2 = ChannelAttention(64)
            self.spatial_attention_2 = SpatialAttention(kernel_size=7)

        # 全连接层
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=576,
                out_features=256
            ),
            nn.Dropout(p=0.4),
            nn.Linear(
                in_features=256,
                out_features=128
            ),
            nn.Dropout(p=0.4),
            nn.Linear(
                in_features=128,
                out_features=num_classes
            )
            # pytorch交叉熵损失函数是混合了softmax的。不需要再使用softmax
        )

    def forward(self, x):
        y = self.block_1_3D(x)
        y = y.view(-1, y.shape[1] * y.shape[2], y.shape[3], y.shape[4])

        if self.self_attention:
            y = self.channel_attention_1(y) * y
            y = self.spatial_attention_1(y) * y

        y = self.block_2_2D(y)
        if self.self_attention:
            y = self.channel_attention_2(y) * y
            y = self.spatial_attention_2(y) * y

        y = y.view(y.size(0), -1)

        y = self.classifier(y)
        return y


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def chuli(img_path, net, patch_size):

    X = sio.loadmat(img_path)[os.path.split(img_path)[-1][:-4]]

    X = np.array(X).reshape(X.shape[0], X.shape[1], X.shape[2])

    y = sio.loadmat(r'D:\software engineering\deeplearning\two\dataset\paviaugthree_gt.mat')['paviaugtthree']

    height = X.shape[0]
    width = X.shape[1]

    #X = applyPCA(X, numComponents=pca_components)

    X = padWithZeros(X, patch_size // 2)

    # 逐像素预测类别
    outputs = np.zeros((height, width))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(height):
        for j in range(width):
            if int(y[i, j]) == 0:
                continue
            else:
                image_patch = X[i:i + patch_size, j:j + patch_size, :]
                image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],1)
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)

                prediction = net(X_test_image)  # 对这张图片的每一个像素点进行特征预测

                prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                outputs[i][j] = prediction + 1 # 得到每一个像素点的分类特征值


    #predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(7, 7))

    spectral.save_rgb(r'D:\software engineering\deeplearning\two\picture\two_net_2_gt.jpg', outputs.astype(int),
                      colors=spectral.spy_colors)

def load_net():
    net = HybridSN()
    net = torch.load(r'D:\software engineering\deeplearning\two\model\two_net_1.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    return net

if __name__ == '__main__':

    img_path = r'D:\software engineering\deeplearning\two\dataset\paviaunewone.mat'

    # 每个像素周围提取 patch 的尺寸
    patch_size = 11

    # 使用 PCA 降维，得到主成分的数量
    # pca_components = 20

    chuli(img_path, load_net(), patch_size)