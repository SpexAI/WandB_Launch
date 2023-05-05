
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import wandb
import os 
import matplotlib.pyplot as plt
import random
import cv2
from detectron2.utils.events import (
    EventWriter,
    get_event_storage,
)

'''
Load description for cocodataset
Input: 
+ project_name: name for project
+ annotation_path: path for annotation in xywh format
+ images_path: image data directory for training data
Output:
+ flower_metadata:
+ dataset_dicts:
'''

def load_coco_data_description(
    project_name = "flower_detection",
    annotation_path = "",
    images_path = ""
):

    if not os.path.isdir(annotation_path):

        print("Invalid annotation path")

        return None, None

    elif not os.path.isdir(images_path):

        print("Invalid images path")

        return None, None
    
    try:

        register_coco_instances(project_name, {}, annotation_path, images_path)
        flower_metadata = MetadataCatalog.get(project_name)
        dataset_dicts = DatasetCatalog.get(project_name)

        return flower_metadata, dataset_dicts
    
    except Exception as error:
        
        print('Raise this error: ' + repr(error))
        
        return None, None
    

'''
Visualisation utility for visualize data:
Input:
+ dataset_dict: dataset dictionary
+ dataset_metadata: dataset metadata
+ scale: Scale zoom rate for the data viewing
'''

def visualize_util(
    dataset_dicts, 
    dataset_metadata, 
    scale = 0.2):
    i = 0
    for d in random.sample(dataset_dicts, 3):
        i+=1
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata = dataset_metadata, scale = scale)
        vis = visualizer.draw_dataset_dict(d)
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.imsave('../images/random_sample{}.jpg'.format(i),vis.get_image()[:, :, ::-1])

class WAndBWriter(EventWriter):
    
    """
    Write all scalars to a wandb tool.
    """

    def __init__(self, window_size: int = 20):

        self._window_size = window_size

    def write(self):

        storage = get_event_storage()
        stats = {}

        for k, v in storage.latest_with_smoothing_hint(self._window_size).items():

            stats[k.replace("/", "-")] = v[0]

        wandb.log(stats, step=storage.iter)

        if len(storage._vis_data) >= 1:

            for img_name, img, step_num in storage._vis_data:

                self._writer.add_image(img_name, img, step_num)

            storage.clear_images()

    def close(self):
        pass

