import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

import os

# 计算HSV直方图
def compute_hsv_histogram(image, bins=(18, 8, 8)):    
    # 将归一化后的像素值恢复到[0, 255]范围
    image = (image * 255).astype(np.uint8)
    # 转换颜色空间 (torch默认读取图像为RGB格式)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]) # 计算图像的HSV颜色直方图

    # hist = cv2.normalize(hist, hist)
    hist = cv2.normalize(hist, hist).flatten() # 标准化以便后续距离度量；展平为一维数组以便调用聚类算法
    return hist

# # 计算RGB直方图
# def compute_rgb_histogram(image, bins=(16, 16, 16)):
#     # 计算直方图
#     hist = cv2.calcHist([image * 255], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256]) # 因为ToTensor对像素值范围进行了归一化，所以我们需要对参数的范围也进行修改
#     print(hist)
#     os._exit(0)
#     # 归一化
#     hist = cv2.normalize(hist, hist).flatten()
#     return hist


def extract_features(loader):
    features = []
    original_images = []
    for batch_idx, (inputs, targets) in enumerate(loader):
        # 将输入图像inputs的维度从PyTorch的(batch_size, channels, height, width)格式 (128, 3, 32, 32)
        # 调整为OpenCV的(batch_size, height, width, channels)的顺序                 (128, 32, 32, 3)
        images = inputs.permute(0, 2, 3, 1).numpy()
        for img in images:
            hist = compute_hsv_histogram(img)
            features.append(hist)
            original_images.append(img)  # 保存原始图像
    return np.array(features), np.array(original_images)

def main():
    # 1. 下载和预处理 CIFAR-10 数据集
    transform = transforms.Compose([
        transforms.ToTensor(), # 将PIL图像转换为PyTorch张量，并对RGB像素值的范围进行了归一化，[0, 255] → [0.0, 1.0]
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    # 2. 提取特征和原始图像
    train_features, train_images = extract_features(trainloader)
    test_features, test_images = extract_features(testloader)
    # print(train_features.shape)
    # print(test_features.shape)

    all_features = np.vstack((train_features, test_features))
    all_images = np.vstack((train_images, test_images))  # 保存所有原始图像

    # 3. K-Means 聚类
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)  # 固定随机种子
    clusters = kmeans.fit_predict(all_features)

    # 计算每个聚类标签的样本数量
    sample_counts = np.bincount(clusters)
    for cluster_label, count in enumerate(sample_counts):
        print(f'Cluster {cluster_label}: {count} samples')

    # 保存原始图像到文件夹
    output_dir = './clustered_images'
    os.makedirs(output_dir, exist_ok=True)

    for idx, img in enumerate(all_images):
        img = (img * 255).astype(np.uint8)  # 将像素值转换回[0, 255]范围
        cv2.imwrite(os.path.join(output_dir, f'{idx}.jpg'), img)

    # 保存聚类结果到CSV文件
    indices = np.arange(len(all_images))
    df = pd.DataFrame({'index': indices, 'label': clusters})
    df.to_csv('clustered_images.csv', index=False)

    # # 5. 显示聚类样本图像
    # for cluster in range(n_clusters):
    #     plt.figure(figsize=(10, 5))
    #     plt.title(f'Cluster {cluster}')
    #     for i in range(9):
    #         idx = np.where(clusters == cluster)[0][i]
    #         img = all_images[idx]  # 使用原始图像
    #         plt.subplot(3, 3, i + 1)
    #         plt.imshow(img)
    #         plt.axis('off')
    #     plt.show()

    # # 6. 创建包含原始图像、索引和聚类标签的数据集
    # original_images_with_labels = []
    # for idx in range(len(all_images)):
    #     # original_images_with_labels.append((idx, all_images[idx], clusters[idx]))
    #     original_images_with_labels.append((idx, clusters[idx]))

    # # 创建DataFrame
    # df = pd.DataFrame(original_images_with_labels, columns=['Index', 'Cluster_Label'])
    
    # # 保存为CSV文件（或其他格式）
    # df.to_csv('clustered_images.csv', index=False)

if __name__ == '__main__':
    main()