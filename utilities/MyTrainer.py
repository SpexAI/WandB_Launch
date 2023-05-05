import os
from detectron2.engine import DefaultTrainer
from detectron2.structures import (
    BitMasks,
    Instances,
    Boxes,
)

import deeplake
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import logging
from detectron2.evaluation.evaluator import DatasetEvaluator
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class BBox_Evalutator(DatasetEvaluator):

    def __init__(
            self,
            distributed=True,
            output_dir=None,
            *,
            use_fast_impl=True,
    ):

        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        self.metric = MeanAveragePrecision()

        self._cpu_device = torch.device("cpu")

    def reset(self):
        self.metric = MeanAveragePrecision()

    def process(self, inputs, outputs):

        for input, output in zip(inputs, outputs):

            preds = [
                dict(
                    boxes=output['instances']._fields['pred_boxes'].tensor.cpu(),
                    labels=output['instances']._fields['pred_classes'].cpu(),
                    scores=output['instances']._fields['scores'].cpu()
                )
            ]
            target = [
                dict(
                    boxes=input['instances']._fields['gt_boxes'].tensor.cpu(),
                    labels=input['instances']._fields['gt_classes'].cpu()
                )
            ]

            self.metric.update(preds, target)

    def evaluate(self, img_ids=None):
        return self.metric.compute()


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return BBox_Evalutator(dataset_name, output_dir=output_folder)

# Augmentation pipeline using Albumentations, add augmentation to the best of  our knowledge. Pipeline should always be the same afterwards
# Do not resize the images here, as this should happen in the model itself using cfg.config
# tform_train = A.Compose([
#     A.Resize(width=1024, height=1024, p=1.0),
#     ToTensorV2(),   #transpose_mask = True,
# ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels', 'bbox_ids'], min_area=3,
#                             min_visibility=0.6))  # 'label_fields' and 'box_ids' are all the fields that will be cut when a bounding box is cut.

# Augmentation pipeline using Albumentations, add augmentation to the best of  our knowledge. Pipeline should always be the same afterwards
# Do not resize the images here, as this should happen in the model itself using cfg.config

tform_train = A.Compose([
    A.LongestMaxSize(max_size=1024, always_apply=True),
    A.Equalize(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.15),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomRotate90(p=0.2),
    A.HorizontalFlip(p=0.2),
    A.VerticalFlip(p=0.2),
    A.RandomGamma(p=0.2),
    A.ChannelShuffle(p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.1),
    ToTensorV2(always_apply=True),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels', 'bbox_ids'], min_area=3,
min_visibility=0.6))
# 'label_fields' and 'box_ids' are all the fields that will be cut when a bounding box is cut.
# Transformation function for pre-processing the deeplake sample before sending it to the model

def transform(sample_in, image_type="images"):
    boxes = sample_in['boxes']
    # Convert any grayscale images to RGB
    images = sample_in[image_type]
    if images.shape[2] == 1:
        images = np.repeat(images, int(3 / images.shape[2]), axis=2)

    # Pass all data to the Albumentations transformation
    # Mask must be converted to a list
    try:
        transformed = tform_train(image=images,
                                  masks=[sample_in['masks'][:, :, i].astype(np.uint8) for i in
                                         range(sample_in['masks'].shape[2])],
                                  bboxes=boxes,
                                  bbox_ids=np.arange(boxes.shape[0]),
                                  class_labels=sample_in['labels'],
                                  )
    except Exception as e:
        print(e)
        return None

    # Convert boxes and labels from lists to torch tensors, because Albumentations does not do that automatically.
    # Be very careful with rounding and casting to integers, because that can create bounding boxes with invalid dimensions
    labels_torch = torch.tensor(transformed['class_labels'], dtype=torch.long)

    boxes_torch = torch.zeros((len(transformed['bboxes']), 4), dtype=torch.int64)
    for b, box in enumerate(transformed['bboxes']):
        boxes_torch[b, :] = torch.tensor(np.round(box))

    # Filter out the masks that were dropped by filtering of bounding box area and visibility
    masks_torch = torch.zeros(
        (len(transformed['bbox_ids']), transformed['image'].shape[1], transformed['image'].shape[2]), dtype=torch.int64)
    if len(transformed['bbox_ids']) > 0:
        masks_torch = torch.tensor(np.stack([transformed['masks'][i] for i in transformed['bbox_ids']], axis=0),
                                   dtype=torch.uint8)
    else:
        print('No bounding boxes left after filtering')

    instances = Instances((transformed['image'].shape[1], transformed['image'].shape[2]))
    instances.gt_masks = BitMasks(masks_torch)
    instances.gt_boxes = Boxes(boxes_torch)
    instances.gt_classes = labels_torch

    # tranform everything to the object Detectron2 expects for training the model
    target = {'image': transformed['image'], 'instances':instances, 'height': 1024, 'width': 1024}

    return target

def collate_fn(batch):
    return list(batch)

def cycle(iterable):
    ### cycle('ABCD') --> A B C D A B C D A B C D ...
    ## creates an infite iterator in case the dataset is not large enough
    while True:
        for x in iterable:
            yield x

def tranform_wrapper(image_type='images'):
    def wrapped_transform(sample_in):
        return transform(sample_in, image_type=image_type)
    return wrapped_transform

class Trainer_Lake(DefaultTrainer):

    @classmethod
    def build_test_loader(cls, cfg):
        ds = deeplake.load(cfg.test_set)
        test_iterator = ds.pytorch(num_workers=cfg.DATALOADER.NUM_WORKERS, batch_size=1, shuffle=False
                                     , transform=tranform_wrapper(image_type=cfg.INPUT.IMAGE_TYPE), collate_fn=collate_fn,
                                     tensors=[cfg.INPUT.IMAGE_TYPE, 'masks', 'boxes', 'labels'], drop_last=True,
                                      decode_method={cfg.INPUT.IMAGE_TYPE: 'numpy'})
        print(f'Using {cfg.test_set} for testing with {len(ds)} samples')
        return iter(cycle(test_iterator))

    @classmethod
    def build_train_loader(cls, cfg):
        ds = deeplake.load(cfg.train_set)
        train_iterator = ds.pytorch(num_workers=cfg.DATALOADER.NUM_WORKERS, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False
                             , transform=tranform_wrapper(image_type=cfg.INPUT.IMAGE_TYPE), collate_fn=collate_fn,
                             tensors=[cfg.INPUT.IMAGE_TYPE, 'masks', 'boxes', 'labels'], decode_method={cfg.INPUT.IMAGE_TYPE: 'numpy'})
        print(f'Using {cfg.train_set} for training with {len(ds)} samples')
        return iter(
            cycle(train_iterator))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return build_evaluator(cfg, dataset_name, output_folder)
