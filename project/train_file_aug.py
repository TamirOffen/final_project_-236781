### imports ###
import json
from pycocotools.coco import COCO
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import os
import numpy as np
import torch
from PIL import Image, ExifTags
from pycocotools.coco import COCO
import transforms as T
import pickle
from torch.utils.tensorboard import SummaryWriter
import random

seed = 211621479 + 941190845
torch.manual_seed(seed)

### dataset ###
# new_annotations_path = '/home/tamiroffen/mini_project/project/new_annotations.json'
cwd = os.getcwd()
new_annotations_path = cwd + '/new_annotations.json'
coco = COCO(new_annotations_path) # Loads dataset as a coco object
with open(new_annotations_path, 'r') as f:
    dataset = json.loads(f.read())  
# load all img paths:
imgs_path = []
for i in dataset['images']:
    imgs_path.append(i['file_name'])
    
### TacoDataset ###
class TacoDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, imgs):
        self.root = root
        self.transforms = transforms
        self.imgs = []  # A list to store all image paths
        for img in imgs:
            self.imgs.append(os.path.join(root, "", img))
            
    def __getitem__(self, img_id):
        # load images 
        img_path = self.imgs[img_id]
        I = Image.open(img_path)
        
        # image rotation
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        if I._getexif():
            exif = dict(I._getexif().items())
            # Rotate portrait and upside down images if necessary
            if orientation in exif:
                if exif[orientation] == 3:
                    I = I.rotate(180,expand=True)
                if exif[orientation] == 6:
                    I = I.rotate(270,expand=True)
                if exif[orientation] == 8:
                    I = I.rotate(90,expand=True)
                    
        
        # coco = COCO(new_annotations_path) not needed anymore, added when loading dataset
        annIds = coco.getAnnIds(imgIds=img_id, catIds=[]) #ids of the anns corresponding w/ img_id
        anns_sel = coco.loadAnns(annIds) #retrieves corresponding ann data
        
        # target:
        boxes = torch.empty(0, 4, dtype=torch.float32)
        labels_list = []
        areas_list = []
        iscrowd_list = []
        for ann in anns_sel:
            # boxes:
            bbox = torch.tensor(ann['bbox'], dtype=torch.float32)
            w, h = bbox[2], bbox[3]
            bbox[2] = bbox[0] + w  # x2
            bbox[3] = bbox[1] + h  # y2
            box = torch.reshape(bbox, (1, 4))
            boxes = torch.cat((boxes, box), dim=0)

            # labels:
            labels_list.append(ann['category_id'])

            # areas:
            areas_list.append(ann['area'])

            # iscrowd:
            iscrowd_list.append(ann['iscrowd'])

        labels = torch.tensor(labels_list, dtype=torch.int64)
        areas = torch.tensor(areas_list)
        iscrowd_tensor = torch.tensor(iscrowd_list, dtype=torch.uint8)
        # img_id_tensor = torch.tensor([img_id])
        img_id_tensor = int(img_id)
            
        target = {}
        target["boxes"] = boxes
        target["iscrowd"] = iscrowd_tensor
        target["image_id"] = img_id_tensor
        target["labels"] = labels
        target["area"] = areas       

        if self.transforms is not None:
            I, target = self.transforms(I, target)

        return I, target
    
    def __len__(self):
        return len(self.imgs)
    
### model ###
def get_model(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # replace the classifier with a new one, that has
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

### data augmentation ###
def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        if random.random() < 0.35: # apply RandomIoUCrop with a p=35
            transforms.append(T.RandomIoUCrop())
        transforms.append(T.RandomZoomOut(fill=[0, 0, 0], side_range=(1.0, 4.0), p=0.5))
        transforms.append(T.RandomPhotometricDistort(p=0.5))
        transforms.append(T.ScaleJitter(target_size=(256, 256), scale_range=(0.5, 2.0)))
    return T.Compose(transforms)

## train loop ###
def main():
    print("training started: rcnn with data augs.") #update
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'using {device} to train')
    
    # Define a directory to save checkpoints
    # checkpoint_dir = '/home/tamiroffen/mini_project/project/checkpoints/rcnn_with_aug'
    checkpoint_dir = cwd + '/checkpoints/rcnn_no_aug'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # our dataset has two classes only - background and person
    num_classes = 7+1
    # use our dataset and defined transformations
    dataset_path = '/datasets/TACO-master/data'
    dataset = TacoDataset(dataset_path, get_transform(train=True), imgs_path) 
    dataset_test = TacoDataset(dataset_path, get_transform(train=False), imgs_path)

    # split the dataset in train and test set (80 20 split)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-300])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-300:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model(num_classes)

    # move model to the right device
    model.to(device)
    
    # original hyperparams:
    # lr = 0.005
    # momentum=0.9
    # weight_decay=0.0005
    # step_size = 3
    # gamma=0.1
 
    # construct an optimizer 
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                 momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=5,
                                                    gamma=0.5)
    
    #optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)

    # for printing (replaced with tensorboard)
    losses = []
    losses_classifier = []
    losses_box_reg = []
    losses_objectness = []
    losses_rpn_box_reg = []
    losses_valid = []
    
    # tensorboard
    writer = SummaryWriter("/home/tamiroffen/mini_project/project/runs/rcnn_with_aug")
    step = 0
    
    num_epochs = 30 #TODO update
    for epoch in range(num_epochs):
        
        train_metrics = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        writer.add_scalar('Overall Training Loss', train_metrics.loss.global_avg, global_step=step)
        writer.add_scalar('Training Loss - classifier', train_metrics.loss_classifier.global_avg, global_step=step)
        writer.add_scalar('Training Loss - Box Reg', train_metrics.loss_box_reg.global_avg, global_step=step)
        writer.add_scalar('Training Loss - Objectness', train_metrics.loss_objectness.global_avg, global_step=step)
        writer.add_scalar('Training Loss - rpn Box Reg', train_metrics.loss_rpn_box_reg.global_avg, global_step=step)
        step += 1
        
        losses.append(train_metrics.loss.avg) 
        losses_classifier.append(train_metrics.loss_classifier.avg)
        losses_box_reg.append(train_metrics.loss_box_reg.avg)
        losses_objectness.append(train_metrics.loss_objectness.avg)
        losses_rpn_box_reg.append(train_metrics.loss_rpn_box_reg.avg)
        
        # Save a checkpoint at the end of every epoch
        checkpoint_filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')

        # Save the checkpoint file into the directory
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_metrics.loss.avg,                                      
        'lr_scheduler_state_dict': lr_scheduler.state_dict()
        }, checkpoint_filename)

        print(f"\n*** Saved checkpoint {checkpoint_filename}")
        
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate_metrics = evaluate(model, data_loader_test, device=device)
        losses_valid.append(evaluate_metrics) # We are appending a CocoEvaluation Object
        
    data_dict = {
        'losses': losses,
        'losses_classifier': losses_classifier,
        'losses_box_reg': losses_box_reg,
        'losses_objectness': losses_objectness,
        'losses_rpn_box_reg': losses_rpn_box_reg,
        'losses_valid': losses_valid
    }
    with open('losses_data.pkl', 'wb') as file:
        pickle.dump(data_dict, file)

    writer.close()
    print("Training is Done!")
    
if __name__ == "__main__":
    main()
    