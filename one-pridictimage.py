import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import spectral
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


# 定义高光谱分类模型
class HybridSN(nn.Module):
    def __init__(self):
        super(HybridSN, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0)
        self.conv4 = nn.Conv2d(576, 64, kernel_size=(3, 3), stride=1, padding=0)
        # attention parameter
        self.conv_phi = nn.Conv2d(576, 576, kernel_size=(1, 1), stride=1, padding=0)
        self.conv_theta = nn.Conv2d(576, 576, kernel_size=(1, 1), stride=1, padding=0)
        self.conv_g = nn.Conv2d(576, 576, kernel_size=(1, 1), stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(576, 576, kernel_size=(1, 1), stride=1, padding=0)

        self.fc1 = nn.Linear(18496, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 16)
        self.dropout = nn.Dropout(0.4)

    def attention(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # print(x_phi.size(), x_theta.size(), x_g.size())
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, c, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1, 19, 19)

        x = self.attention(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

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

    #传入的数据集为降了维之后再进行切割之后的
    X = sio.loadmat(img_path)[os.path.split(img_path)[-1][:-4]]

    #这里传入的标签数据集为切割好了的
    y = sio.loadmat(r'D:\software engineering\deeplearning\two\dataset\Indian_pines_gt.mat')['indian_pines_gt']

    height = X.shape[0]
    width = X.shape[1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # X = applyPCA(X, numComponents=pca_components)

    X = padWithZeros(X, patch_size // 2)

    # 逐像素预测类别
    outputs = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if int(y[i, j]) == 0:
                continue
            else:
                image_patch = X[i:i + patch_size, j:j + patch_size, :]
                image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                                  1)
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)

                prediction = net(X_test_image)
                prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                outputs[i][j] = prediction + 1

    spectral.save_rgb(r'D:\software engineering\deeplearning\two\picture\first_net_2_gt.jpg', outputs.astype(int), colors=spectral.spy_colors)#保存分类标签


def load_net():
    net = HybridSN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.load(r'D:\software engineering\deeplearning\two\model\first_net_1.pth')
    net = net.to(device)
    return net

if __name__ == '__main__':

    # 每个像素周围提取 patch 的尺寸
    patch_size = 25

    # 使用 PCA 降维，得到主成分的数量
    # pca_components = 30

    img_path = r'D:\software engineering\deeplearning\two\dataset\indian_pines_correctedone.mat'

    chuli(img_path, load_net(), patch_size)