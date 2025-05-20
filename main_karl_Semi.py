# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
# yash92328_ai4mars_terrainaware_autonomous_driving_on_mars_path = kagglehub.dataset_download('yash92328/ai4mars-terrainaware-autonomous-driving-on-mars')
# print('Data source imported to path:', yash92328_ai4mars_terrainaware_autonomous_driving_on_mars_path)

# print('Data source import complete.')

# !pip install -q pytorch-lightning torchmetrics wandb
# !mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !git clone https://github.com/Kaggle/docker-python.git
import sys
sys.path.append("./docker-python/patches")
import os
import random
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import ConcatDataset,Subset

import wandb  
# from kaggle_secrets import UserSecretsClient
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Accuracy, ConfusionMatrix, JaccardIndex
# from torcvision import transforms
# from torchvision.transforms import transforms
from tqdm import tqdm

random.seed(42)
# Log into Wandb
# !wandb login ed665cba88a79ecff72b0c76687789842aebfb52
device='cuda' if torch.cuda.is_available() else 'cpu'

import os
import time
cur_time = str(int(time.time()))
print("Current time:", cur_time)

# Paths
IMAGES_PATH = "D:/A_KTH_Course/DL_DataScience/PROJECT-1-A/yash92328/ai4mars-terrainaware-autonomous-driving-on-mars/versions/1/ai4mars-dataset-merged-0.1/msl/images/edr"
MASK_PATH_TRAIN = "D:/A_KTH_Course/DL_DataScience/PROJECT-1-A/yash92328/ai4mars-terrainaware-autonomous-driving-on-mars/versions/1/ai4mars-dataset-merged-0.1/msl/labels/train"
MASK_PATH_TEST = "D:/A_KTH_Course/DL_DataScience/PROJECT-1-A/yash92328/ai4mars-terrainaware-autonomous-driving-on-mars/versions/1/ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min3-100agree/"



class AI4MARSDataset(Dataset):
    def __init__(self, images_path: str, masks_path: str, dataset_size: int = 500, pseudo_labels = None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.dataset_size = dataset_size
        self.pseudo_labels = pseudo_labels or {}

        # 获取所有匹配的图片-mask对（限制总数）
        images = set(os.listdir(images_path))
        self.masks = [mask for mask in os.listdir(masks_path) if mask[:-4] + ".JPG" in images][:dataset_size]

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_name = self.masks[idx]
        image_name = mask_name[:-4] + ".JPG"

        image_path = os.path.join(self.images_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)

        if mask_name in self.pseudo_labels:
            mask = self.pseudo_labels[mask_name]
        else:
            mask_path = os.path.join(self.masks_path, mask_name)
            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
            mask = np.array(mask, dtype=np.uint8)
            mask[mask == 255] = 4
            mask = torch.from_numpy(mask).long()
        weight = self.weights[idx] if hasattr(self, 'weights') else 1.0
        return image, mask, weight
        # return image, mask

class AI4MARSDataModule(pl.LightningDataModule):
    def __init__(self, images_path: str, masks_path: str, dataset_size: int = 5000, batch_size: int = 32, num_workers: int = 0, Semi_supervised=None):
        super().__init__()
        self.images_path = images_path
        self.masks_path = masks_path
        self.dataset_size = dataset_size
        self.unlabel_size = int(dataset_size * float(Semi_supervised['Unlabel']) / 100.0)
        self.label_size = dataset_size - self.unlabel_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.full_dataset = AI4MARSDataset(self.images_path, self.masks_path, self.dataset_size)
 
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 迁移学习：冻结模型的前几层


def custom_freeze(self, n: int, freeze_bn=True):
    total_layers=[]
    for name, _ in self.model.named_parameters():
        total_layers.append(name)
    unfreeze_classifier=1 if -n!=0 else 0

    n-=1# minus the classifier layer
    # print("inters ",(-n)*3-unfreeze_classifier*5-5)
    layers_to_unfreeze=total_layers[(-n)*3-unfreeze_classifier*5-5:-5]  # Get the last n layers to unfreeze, if -2, unfreeze the last layer
    # 3=conv+bn1+bn2
    # +5=classifier0,1,4
    # :5=aux
    for name, param in self.model.named_parameters():
        # If any key in the list matches the parameter name → unfreeze
        if any(key in name for key in layers_to_unfreeze):
            param.requires_grad = True
            # 如果冻结BN层，则只冻结在 layers_to_unfreeze 中的 BN 层
            if freeze_bn and ('bn' in name or 'downsample.1' in name):
                if any(key in name for key in layers_to_unfreeze):  # 保证只有在解冻的层内冻结BN
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            param.requires_grad = False
        # Always freeze BatchNorm if requested
        # for detailed structure see the pic at https://zhuanlan.zhihu.com/p/353235794
     # Stampa una verifica rapida
    # print("\nTrainable params (requires_grad=True)/////////////////////////////////////////////////////////////////////////:")
    # for name, param in self.model.named_parameters():
    #     if param.requires_grad:
    #         print("  •", name)
    # print("\nFreezed params (requires_grad=False):")
    # for name, param in self.model.named_parameters():
    #     if not param.requires_grad:
    #         print("  •", name)

# 迁移学习：冻结模型的前几层
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

class ImageSegmentationModel(pl.LightningModule):
    def __init__(self, num_classes: int = 5, learning_rate: float = 1e-4, Tune=None, Semi_supervised=None):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.model_weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
        self.model = torchvision.models.segmentation.fcn_resnet50(weights=self.model_weights)
        self.model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

        self.loss = nn.CrossEntropyLoss()
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.iou = JaccardIndex(task='multiclass', num_classes=num_classes)
        self.max_Weight = Semi_supervised['max_weight'] if Semi_supervised else 0.5
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.freeze_to = Tune['freeze_layers']
        self.freeze_bn = Tune['freeze_bn']
        if self.freeze_to > 0:
            print(f"Freezing model layers to: {-1 * self.freeze_to}")
            custom_freeze(self, n=self.freeze_to, freeze_bn=self.freeze_bn)
            #move to GPU
            self.model.to(device)
        else:
            print("No layers unfrozen~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    def forward(self, x):
        return self.model(x)['out']

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        preds = self(x)
        loss = self.loss(preds, y)

        # 动态调整权重（假设最大为 0.5）
        current_weight = self.max_Weight * (self.current_epoch / self.trainer.max_epochs)

        weighted_loss = loss * w.to(loss.device) * current_weight
        self.log('train_loss', weighted_loss.mean())
        return weighted_loss.mean()

        

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        preds = self(x)
        loss = self.loss(preds, y)
        preds = torch.argmax(preds, dim=1)

        self.log('val_loss', loss, on_step=True, on_epoch=True)
        val_acc = self.accuracy(preds, y)
        self.log('val_acc', val_acc, on_step=True, on_epoch=True)
        self.log('val_iou', self.iou(preds, y), on_step=True, on_epoch=True)

class PseudoLabelDataset(Dataset):
    def __init__(self, pseudo_data):
        self.pseudo_data = pseudo_data  # List of (image_tensor, mask_tensor)

    def __len__(self):
        return len(self.pseudo_data)

    def __getitem__(self, idx):
        image, mask, weight = self.pseudo_data[idx]
        weight = 1.0  # 或者你可以根据置信度设置不同权重
        return image, mask, weight

def train_single_run(data_module: AI4MARSDataModule, epochs: int = 10, path=None, Tune=None, Semi_supervised=None, iterations: int = 3):
    
    # dataset_size = len(data_module.full_dataset)
    labeled_set, unlabeled_set = random_split(data_module.full_dataset, [data_module.label_size, data_module.unlabel_size])
    unlabeled_set.weights= torch.ones(len(unlabeled_set))*0.5
    train_size = int(len(labeled_set) * 0.8)
    val_size = len(labeled_set) - train_size
    train_dataset, val_dataset = random_split(labeled_set, [train_size, val_size])

    model = ImageSegmentationModel(Tune=Tune)
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path))
    model.to(device)

    wandb_logger = pl.loggers.WandbLogger(name="AI4MARS_iterative", project="AI4MARS")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="checkpoint-{epoch:02d}-{val_loss:.4f}", 
        save_top_k=1, 
        mode="min")

    for iteration in range(iterations):
        print(f"================== Iteration {iteration+1}/{iterations} ==================")

        train_dataloader = DataLoader(train_dataset, batch_size=data_module.batch_size, shuffle=True, num_workers=data_module.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=data_module.batch_size, num_workers=data_module.num_workers)

        # pseudo_dataloader = DataLoader(unlabeled_set, batch_size=data_module.batch_size, shuffle=True, num_workers=data_module.num_workers)

        trainer = pl.Trainer(
            max_epochs=epochs,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            accelerator='gpu',
            devices=-1,
            log_every_n_steps=50
        )
        trainer.fit(model, train_dataloader, val_dataloader)

        pseudo_threshold=Semi_supervised['pseudo_threshold']
        model.eval()
        model.to(device)
        pseudo_data = []
        # new_pseudo = []
        # pseudo_indices = []
        # with torch.no_grad():
        #     for img, mask, _ in DataLoader(unlabeled_set, batch_size=1):
        #         img = img.to(device)
        #         model.to(device)
        #         output = model(img)#['out']
        #         probs = torch.softmax(output, dim=1)
        #         confidence, preds = probs.max(dim=1)
        #         if confidence.mean() > pseudo_threshold:
        #             pseudo_data.append((img.squeeze(0).cpu(), preds.squeeze(0).cpu()))

        # 假设你有一个函数来计算相似度（例如 Dice 系数）
        def dice_coefficient_multiclass(pred, target, num_classes, smooth=1e-6):
            dice_scores = []
            for c in range(num_classes):
                # 二值化：类别 c 的像素设为 1，其他设为 0
                pred_c = (pred == c).float()
                target_c = (target == c).float()
                
                # 展平
                pred_c = pred_c.contiguous().view(-1)
                target_c = target_c.contiguous().view(-1)
                
                # 计算交集和并集
                intersection = (pred_c * target_c).sum()
                dice = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
                dice_scores.append(dice)
            
            # 返回平均 Dice 系数
            return sum(dice_scores) / len(dice_scores)
        with torch.no_grad():
            for img, mask, _ in DataLoader(unlabeled_set, batch_size=1):
                img = img.to(device)
                mask = mask.to(device)  # 确保真实掩码也在设备上
                # model.to(device)
                output = model(img)  # 假设输出是 logits
                probs = torch.softmax(output, dim=1)
                _, preds = probs.max(dim=1)  # 获取预测掩码

                # 将预测和真实掩码转换为二值化形式（如果需要）
                preds = preds.squeeze(0).float()  # 去掉 batch 维度
                mask = mask.squeeze(0).float()    # 去掉 batch 维度

                # 计算相似度（这里以 Dice 系数为例）
                similarity = dice_coefficient_multiclass(preds, mask, num_classes=5)

                # 判断相似度是否大于阈值
                if similarity > pseudo_threshold:
                    pseudo_data.append((img.squeeze(0).cpu(), preds.cpu(), mask.cpu()))  # 可选：保存掩码用于进一步分析
                    # print(f"Sample included: Dice similarity = {similarity:.4f}")
                # else:
                    # print(f"Sample excluded: Dice similarity = {similarity:.4f}")
        new_pseudo=PseudoLabelDataset(pseudo_data)

        print(f"> Adding {len(new_pseudo)} pseudo-labeled samples to labeled set")
        labeled_set = ConcatDataset([labeled_set, new_pseudo])
        remaining_indices = list(set(range(len(unlabeled_set))) - set(new_pseudo))
        unlabeled_set = Subset(unlabeled_set, remaining_indices)
    cur_time = str(int(time.time()))
    torch.save(model.state_dict(), "final_model_" + cur_time + ".pth")
    torch.save(model.state_dict(), "weit_" + cur_time + ".pth")
    return wandb_logger.experiment.summary, cur_time


# Preprocess images
def preprocess_image(image_path: str, return_tensor: bool = False):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image_normalized = np.asarray(image, dtype=np.float32) / 255.0

    if return_tensor:
        image_normalized = np.transpose(image_normalized, (2, 0, 1))
        image_tensor = torch.from_numpy(image_normalized).unsqueeze(0)
        return image, image_tensor

    return image

# Preprocess masks
def preprocess_mask(mask_path: str):
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    mask[mask == 255] = 4
    return mask

# Find an image in the test set
def display_segmentation(model,index):
    mask_files = [filename for filename in os.listdir(MASK_PATH_TEST) if filename.endswith("_merged.png")]
    test_image_name = mask_files[index][:-11] + ".JPG"

    # Load the test image
    test_image_path = os.path.join(IMAGES_PATH, test_image_name)
    test_image, test_image_tensor = preprocess_image(test_image_path, return_tensor=True)
    test_image_tensor = test_image_tensor.to(device)

    # Perform prediction
    with torch.no_grad():
        prediction = model(test_image_tensor)
        predicted_mask = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()

    # Load the ground truth segmentation
    ground_truth_mask_path = os.path.join(MASK_PATH_TEST, test_image_name[:-4] + "_merged.png")
    ground_truth_mask = preprocess_mask(ground_truth_mask_path)

    return test_image, ground_truth_mask, predicted_mask


class TestDataset(Dataset):
    def __init__(self, images_path: str, masks_path: str):
        self.images_path = images_path
        self.masks_path = masks_path
        self.mask_files = [mask for mask in os.listdir(masks_path) if mask.endswith("_merged.png")]

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        mask_name = self.mask_files[idx]
        image_name = mask_name[:-11] + ".JPG"

        image_path = os.path.join(self.images_path, image_name)
        _, image_tensor = preprocess_image(image_path, return_tensor=True)

        mask_path = os.path.join(self.masks_path, mask_name)
        mask = preprocess_mask(mask_path)

        # Remove the extra dimension
        image_tensor = image_tensor.squeeze(0)

        return image_tensor, mask


def evaluate_test_set(model, images_path, masks_path, batch_size=32, num_workers=2, device='cuda',training_semi_supervised=False):

    # Load dataloaders
    test_dataset = TestDataset(images_path, masks_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    # Initialize arrays for storing ground truth and predictions
    ground_truths = []
    predictions = []

    for batch_images, batch_masks in tqdm(test_dataloader):
        # Move the images to the GPU
        batch_images = batch_images.to(device)

        # Perform prediction
        with torch.no_grad():
            batch_prediction = model(batch_images)
            batch_predicted_masks = torch.argmax(batch_prediction, dim=1).cpu().numpy()

        # Flatten and append the ground truth and predictions
        ground_truths.extend([mask.flatten() for mask in batch_masks])
        predictions.extend([mask.flatten() for mask in batch_predicted_masks])
    
    # Compute accuracy
    accuracy = accuracy_score(np.concatenate(ground_truths), np.concatenate(predictions))

    # Compute IoU (Jaccard score)
    iou = jaccard_score(np.concatenate(ground_truths), np.concatenate(predictions), average="weighted")

    # Compute confusion matrix
    cm = confusion_matrix(np.concatenate(ground_truths), np.concatenate(predictions))
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    return accuracy, iou, cm_normalized





def train_process(Init_params,Tune,Semi_supervised):
    # Unpack parameters
    DATASET_SIZE = Init_params['DATASET_SIZE']
    BATCH_SIZE = Init_params['BATCH_SIZE']
    EPOCHS = Init_params['EPOCHS']
    init_weit_path = Init_params['init_weit_path']
 

    # Load dataset
    weit_path = init_weit_path if init_weit_path else None
    # Train model
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # RANDOM SHUFFLE:
    # for i in range(EPOCHS):
    #     print("Epoch:", i+1)
    #     # Load dataset
    #     rock_data = AI4MARSDataModule(IMAGES_PATH, MASK_PATH_TRAIN, dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE, num_workers=0)
    #     rock_data.setup()
    #     print("Loading model from path:", weit_path)
    #     summary,cur_time = train_single_run(rock_data, epochs=1, path=weit_path, freeze_to=freeze_to, freeze_bn=freeze_bn)
    #     path = "weit_"+str(cur_time)+".pth"
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #FIXED 1000 DATA:
    rock_data = AI4MARSDataModule(IMAGES_PATH, MASK_PATH_TRAIN, dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE, num_workers=0,Semi_supervised=Semi_supervised)
    rock_data.setup()
    summary,cur_time = train_single_run(rock_data, epochs=EPOCHS, path=weit_path, Tune=Tune, Semi_supervised=Semi_supervised)
    path = "weit_"+str(cur_time)+".pth"
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    print("Training summary:", summary)
    print("Model saved at:", path)
    
    # End current run
    wandb.finish()
    return path


def test_process(weit_to_load='weit_11.pth',Tune=None):
    # Load the saved model
    path = weit_to_load
    model = ImageSegmentationModel(Tune=Tune)
    model.load_state_dict(torch.load(path))
    model.eval()
    model.to(device)

    # Choose indices#这里是要显示的图片索引
    indices = [0, 50, 100]
    segmentations = [display_segmentation(model,index) for index in indices]

    # Plot the images
    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 15))
    for i, (test_image, ground_truth_mask, predicted_mask) in enumerate(segmentations):
        axes[i, 0].imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title("Original image")
        axes[i, 1].imshow(ground_truth_mask, cmap="inferno")
        axes[i, 1].set_title("Crowdsourced segmentation")
        axes[i, 2].imshow(predicted_mask, cmap="inferno")
        axes[i, 2].set_title("Predicted segmentation")

    plt.tight_layout()
    # Evaluate the model
    accuracy, iou, cm_normalized = evaluate_test_set(model, IMAGES_PATH, MASK_PATH_TEST, device=device)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"IoU: {iou:.4f}")

    # Plot the normalized confusion matrix using Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".4f")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.show()

if __name__ == '__main__':
    weits=['weit_COCO.pth', 
           'weit_11.pth',
           'final_model_1747331987.pth',
           '']
    
    weit_to_load=weits[1]

    Init_params={
        'DATASET_SIZE': 200,
        'BATCH_SIZE': 32,
        'EPOCHS': 3, 
        'init_weit_path': weit_to_load
    }

    # Hyperparameters
    Tune={
        'freeze_layers': 30,
        'freeze_bn': False
    }
    Semi_supervised={
        'Unlabel': 0,#50% unlabelled data
        'max_weight': 1.0,#maximum weight for unlabelled data in loss
        'pseudo_threshold': 0.9#confidence threshold for pseudo-labeling
    }


    for i in [0,50,90,99]:
        print("Percentage:"+str(i)+"///////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
        Semi_supervised['Unlabel']=i
        #TRAINING
        weit_to_load=train_process(Init_params,Tune,Semi_supervised)
        
        #TESTING
        test_process(weit_to_load,Tune)
    # test_process(weit_to_load,Tune)
    
