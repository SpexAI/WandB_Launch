from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.engine import (
    hooks, 
)
from utilities.MyTrainer import Trainer_Lake
from utilities.wandb_writer import WAndBWriter
from utilities.wandb_log import log_segmentation_results
from detectron2 import model_zoo
import yaml
import wandb
import os
import deeplake
import json


def main():
    ### LOADING TRAINING CONFIG FILE ###
    print("\nLoading configuration for training\n")
    f = open("./config/train.json")
    json_config = json.load(f)

    # init when run standalone
    print('INIT PROJECT:', json_config["project_name"])
    run = wandb.init(project=json_config["project_name"], job_type="train")


    ### WANDB CONFIG ###
    wandb.config.config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    wandb.config.num_workers = json_config['num_workers']
    wandb.config.mask_format = "bitmask"
    wandb.config.score_thresh_test = 0.4
    wandb.config.im_size = 1024
    wandb.config.lr_scheduler_name = "WarmupMultiStepLR"
    
    # Config when run standalone
    wandb.config.epochs = 100
    wandb.config.batch_size_per_image = 256
    wandb.config.base_lr = 0.001
    wandb.config.image_type = 'images'
    wandb.config.ims_per_batch = 2

    ### LOAD JSON CONFIG ###
    bucket_path = json_config["bucket_path"]
    test_path = json_config["test_path"]
    vis_samples = json_config["vis_samples"]

    ### LOAD DETECTRON CONFIG ###
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN ='' 
    cfg.DATASETS.TEST = '' 

    cfg.train_set = 'hub://spexai/testRGB_test'
    cfg.val_set = 'hub://spexai/testRGB_test'
    cfg.test_set = 'hub://spexai/testRGB_test'

    cfg.credentials = json_config["credentials"]
    cfg.access_key = json_config["access_key"]

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = wandb.config.num_workers

    # Solver
    cfg.SOLVER.LR_SCHEDULER_NAME = wandb.config.lr_scheduler_name
    cfg.SOLVER.IMS_PER_BATCH = wandb.config.ims_per_batch
    cfg.SOLVER.BASE_LR = wandb.config.base_lr
    cfg.SOLVER.MAX_ITER = wandb.config.epochs
    cfg.SOLVER_CHECKPOINT_PERIOD = 5

    # Input
    cfg.INPUT.MASK_FORMAT = wandb.config.mask_format
    cfg.INPUT.MASK_ON = True
    
    # Model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(wandb.config.config_file)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = wandb.config.batch_size_per_image 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = wandb.config.score_thresh_test
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 class (flower)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1

    # Load mean and std
    train_ds = deeplake.load(cfg.train_set)

    # Detectron2 Augmentation
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT_RANDOM_FLIP = "horizontal"
    cfg.INPUT.MIN_SIZE_TRAIN = (wandb.config.im_size,) # just train on small set for now
    cfg.INPUT.IMAGE_TYPE = wandb.config.image_type
    
    # Test
    cfg.TEST_EVAL_PERIOD = 5
    cfg.TEST.DETECTIONS_PER_IMAGE = 100

    ## TRAINING ###
    run.log_code()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer_Lake(cfg) ## Use the overwritten trainer to utilize the datalake data
    trainer.resume_or_load(resume=False)
    trainer.register_hooks(
        [
            hooks.PeriodicWriter([WAndBWriter()])
        ])
    cfg_wandb = yaml.safe_load(cfg.dump())
    wandb.config.update(cfg_wandb)
    trainer.train()

    ### STORING TO WANDB ###
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = wandb.config.score_thresh_test
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    ds = deeplake.load(test_path)
    log_segmentation_results(predictor, ds, vis_samples)
    
    # Save artifacts
    try:
        model_data = wandb.Artifact(json_config["wandb_test"], type="model") ## Update this to something meaningful, or model name
        model_data.add_file(os.path.join(cfg.OUTPUT_DIR, json_config["model_path"]))
        run.log_artifact(model_data)

    except Exception as error:
        print('Raise this error: ' + repr(error))
    run.finish()

if __name__ == "__main__":
    json_file = open("./config/train.json")
    json_config_file = json.load(json_file)
    print('Project:', json_config_file["project_name"])
    main()



