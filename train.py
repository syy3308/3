import torch
from torch.utils.data import DataLoader, Dataset
from model import YOLOWithAttention
from loss import DetectionLoss
from backbone import ResNetBackbone
import cv2
import os
import json

# 1. 数据集类定义
class CrowdHumanDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.annotations = self.load_annotations(annotation_file)
        self.transform = transform

    def load_annotations(self, file_path):
        annotations = []
        with open(file_path, 'r') as f:
            for line in f:
                annotations.append(json.loads(line.strip()))
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        img_path = os.path.join(self.image_dir, anno['ID'] + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes = anno['gtboxes']
        if self.transform:
            img = self.transform(img)
        # 这里需要将 bboxes 转换为适合训练的格式
        targets = self.process_bboxes(bboxes)
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1), targets

    def process_bboxes(self, bboxes):
        # 将 bboxes 转换为适合模型的格式
        conf_targets = []
        loc_targets = []
        for box in bboxes:
            if 'hbox' in box:  # 头部框
                conf_targets.append(1)  # 假设 1 是头部类别
                loc_targets.append(box['hbox'])
            elif 'fbox' in box:  # 全身框
                conf_targets.append(2)  # 假设 2 是全身类别
                loc_targets.append(box['fbox'])
        return {'conf': torch.tensor(conf_targets, dtype=torch.float32),
                'loc': torch.tensor(loc_targets, dtype=torch.float32)}

# 2. 数据加载
def get_dataloader(image_dir, annotation_file, batch_size=4, shuffle=True):
    dataset = CrowdHumanDataset(image_dir=image_dir, annotation_file=annotation_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)
    return dataloader

# 3. 训练函数定义
def train_model(data_loader, model, optimizer, criterion, num_epochs=250, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            images, targets = zip(*batch)
            images = torch.stack(images).to(device)
            conf_targets = torch.cat([t['conf'] for t in targets]).to(device)
            loc_targets = torch.cat([t['loc'] for t in targets]).to(device)

            optimizer.zero_grad()
            conf_preds, loc_preds = model(images)
            loss = criterion(conf_preds, loc_preds, conf_targets, loc_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

# 4. 主程序
if __name__ == "__main__":
    # 数据加载路径
    train_image_dir = "D:/ProgramData/PyCharm Community Edition 2024.3.5/PycharmProjects/PythonProject2/CrowdHuman/CrowdHuman_train01"
    train_annotation_file = "D:/ProgramData/PyCharm Community Edition 2024.3.5/PycharmProjects/PythonProject2/CrowdHuman/annotation_train.odgt"

    # 加载数据
    train_loader = get_dataloader(image_dir=train_image_dir, annotation_file=train_annotation_file)

    # 模型初始化
    backbone = ResNetBackbone()
    model = YOLOWithAttention(backbone=backbone, num_classes=2)

    # 损失函数和优化器
    criterion = DetectionLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 开始训练
    train_model(train_loader, model, optimizer, criterion)