""""
Unet分割
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

# Your UNetBasedModel import (assuming it's in a separate file or same directory)
# from Uet import UNetBasedModel
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x

class UNetEncoder(nn.Module):
    def __init__(self, n_channels=3):
        super(UNetEncoder, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

class UNetBasedModel(nn.Module):
    def __init__(self):
        super(UNetBasedModel, self).__init__()    
        self.encoder = UNetEncoder()
        
        # Adjust channel numbers to match original implementation
        self.rfb1 = RFB_modified(64, 64)
        self.rfb2 = RFB_modified(128, 64)
        self.rfb3 = RFB_modified(256, 64)
        self.rfb4 = RFB_modified(512, 64)
        self.rfb5 = RFB_modified(1024, 64)
        
        self.up1 = Up(128, 64)  # 64+64
        self.up2 = Up(128, 64)  # 64+64
        self.up3 = Up(128, 64)  # 64+64
        self.up4 = Up(128, 64)  # 64+64
        
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, x2, x3, x4, x5 = self.encoder(x)
        
        # RFB modules
        x1 = self.rfb1(x1)
        x2 = self.rfb2(x2)
        x3 = self.rfb3(x3)
        x4 = self.rfb4(x4)
        x5 = self.rfb5(x5)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        out1 = F.interpolate(self.side1(x), scale_factor=8, mode='bilinear')
        
        x = self.up2(x, x3)
        out2 = F.interpolate(self.side2(x), scale_factor=4, mode='bilinear')
        
        x = self.up3(x, x2)
        out = F.interpolate(self.head(x), scale_factor=2, mode='bilinear')
        
        return out, out1, out2

# Data transformation classes
class ToTensor(object):
    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': TF.to_tensor(image), 'label': TF.to_tensor(label)}

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {
            'image': TF.resize(image, self.size),
            'label': TF.resize(label, self.size, interpolation=transforms.InterpolationMode.BICUBIC)
        }

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            return {'image': TF.hflip(image), 'label': TF.hflip(label)}
        return {'image': image, 'label': label}

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']
        if random.random() < self.p:
            return {'image': TF.vflip(image), 'label': TF.vflip(label)}
        return {'image': image, 'label': label}

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = TF.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}

# Dataset class for training
class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode='train'):
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image': image, 'label': label}
        data = self.transform(data)
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

# Dataset class for testing
class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(image_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)
        name = self.images[self.index].split('/')[-1]
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

# Metrics calculation
def calculate_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    pred_flat = pred_bin.view(-1)
    target_flat = target_bin.view(-1)
    tn, fp, fn, tp = confusion_matrix(
        target_flat.cpu().numpy(),
        pred_flat.cpu().numpy(),
        labels=[0, 1]
    ).ravel()
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return dice, iou, precision, recall

# Structure loss function
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

# Plotting function
def plot_metrics(epoch_losses, epoch_metrics, save_path):
    os.makedirs(os.path.join(save_path, 'plots'), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='Total Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'plots', 'loss_curve.png'))
    plt.close()
    metrics_names = ['Dice', 'IoU', 'Precision', 'Recall']
    plt.figure(figsize=(15, 10))
    for i, name in enumerate(metrics_names):
        plt.subplot(2, 2, i+1)
        plt.plot([m[i] for m in epoch_metrics], label=name)
        plt.title(f'{name} Curve')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'plots', 'metrics_curve.png'))
    plt.close()

# Main training function
def main(args):
    # Initialize dataset and dataloader
    dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model, optimizer, and scheduler
    # Initialize model, optimizer, and scheduler
    model = UNetBasedModel()
    model.to(device)
    optimizer = torch.optim.AdamW([{"params": model.parameters(), "initial_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1.0e-7)
    
    # model = UNetBasedModel()
    # model.to(device)
    # optim = optim.AdamW([{"params": model.parameters(), "initial_lr": args.lr}], lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = CosineAnnealingLR(optim, args.epoch, eta_min=1.0e-7)
    os.makedirs(args.save_path, exist_ok=True)
    
    # Initialize lists to store metrics
    epoch_losses = []
    epoch_metrics = []
    
    # Training loop
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        epoch_precision = 0.0
        epoch_recall = 0.0
        num_batches = 0
        
        for i, batch in enumerate(dataloader):
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            
            # optim.zero_grad()
            optimizer.zero_grad()  # Change here
            pred0, pred1, pred2 = model(x)
            
            # Calculate losses for all outputs
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            
            loss.backward()
            optimizer.step()  # Change here
            # optim.step()
            
            # Calculate metrics on the final prediction (pred0)
            with torch.no_grad():
                pred_sigmoid = torch.sigmoid(pred0)
                dice, iou, precision, recall = calculate_metrics(pred_sigmoid, target)
                
                epoch_dice += dice
                epoch_iou += iou
                epoch_precision += precision
                epoch_recall += recall
                total_loss += loss.item()
                num_batches += 1
            
            if i % 50 == 0:
                print(f"epoch:{epoch+1}-{i+1}: loss:{loss.item():.4f}, "
                      f"Dice:{dice:.4f}, IoU:{iou:.4f}, "
                      f"Precision:{precision:.4f}, Recall:{recall:.4f}")
        
        # Calculate epoch averages
        avg_loss = total_loss / num_batches
        avg_dice = epoch_dice / num_batches
        avg_iou = epoch_iou / num_batches
        avg_precision = epoch_precision / num_batches
        avg_recall = epoch_recall / num_batches
        
        epoch_losses.append(avg_loss)
        epoch_metrics.append((avg_dice, avg_iou, avg_precision, avg_recall))
        
        print(f"Epoch {epoch+1} Summary - Loss: {avg_loss:.4f}, "
              f"Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, "
              f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")
        
        scheduler.step()
        
        # Save checkpoints
        if (epoch+1) % 5 == 0 or (epoch+1) == args.epoch:
            torch.save(model.state_dict(), os.path.join(args.save_path, f'UNet-{epoch+1}.pth'))
            print(f'[Saving Snapshot:] {os.path.join(args.save_path, f"UNet-{epoch+1}.pth")}')
    
    # Plot metrics after training
    plot_metrics(epoch_losses, epoch_metrics, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("UNet Training")
    parser.add_argument("--train_image_path", type=str, required=True,
                        help="path to the training images")
    parser.add_argument("--train_mask_path", type=str, required=True,
                        help="path to the training masks")
    parser.add_argument('--save_path', type=str, required=True,
                        help="path to store the checkpoints")
    parser.add_argument("--epoch", type=int, default=20,
                        help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=12, help="batch size")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    args = parser.parse_args()
    main(args)