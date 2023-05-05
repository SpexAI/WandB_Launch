import wandb
import numpy as np
from detectron2.utils.visualizer import Visualizer, ColorMode

def _parse_prediction(pred):
    """
    Parse prediction of one image and return the primitive martices to plot wandb media files.
    Moves prediction from GPU to system memory.
    Args:
        pred (detectron2.structures.instances.Instances): Prediction instance for the image
    returns:
        Dict (): parsed predictions
    """
    parsed_pred = {}
    if pred.get("instances") is not None:
        pred_ins = pred["instances"].to("cpu")
        #pred_ins = pred_ins[pred_ins.scores > 0.7]
        parsed_pred["boxes"] = (
            pred_ins.pred_boxes.tensor.tolist() if pred_ins.has("pred_boxes") else None
        )
        parsed_pred["classes"] = (
            pred_ins.pred_classes.tolist() if pred_ins.has("pred_classes") else None
        )
        parsed_pred["scores"] = pred_ins.scores.tolist() if pred_ins.has("scores") else None
        parsed_pred["pred_masks"] = (
            pred_ins.pred_masks.cpu().detach().numpy() if pred_ins.has("pred_masks") else None
        )  # wandb segmentation panel supports np
        parsed_pred["pred_keypoints"] = (
            pred_ins.pred_keypoints.tolist() if pred_ins.has("pred_keypoints") else None
        )

    if pred.get("sem_seg") is not None:
        parsed_pred["sem_mask"] = pred["sem_seg"].argmax(0).cpu().detach().numpy()

    if pred.get("panoptic_seg") is not None:
        # NOTE: handling void labels isn't neat.
        panoptic_mask = pred["panoptic_seg"][0].cpu().detach().numpy()
        # handle void labels( -1 )
        panoptic_mask[panoptic_mask < 0] = 0
        parsed_pred["panoptic_mask"] = panoptic_mask

    return parsed_pred

def _plot_table_row(pred, img, id, detectron_img):
    """
    plot prediction on one image
    Args:
        pred (Dict): Prediction for one image
    """
    classes = ['Flower']
    # Process Bounding box detections
    boxes = {}
    avg_conf_per_class = [0 for i in range(len(classes))]
    counts = {}
    class_labels = {}
    if pred.get("boxes") is not None:
        boxes_data = []
        for i, box in enumerate(pred["boxes"]):
            pred_class = int(pred["classes"][i])
            boxes_data.append(
                {
                    "position": {
                        "minX": box[0],
                        "minY": box[1],
                        "maxX": box[0],
                        "maxY": box[1],
                    },
                    "class_id": i,
                    "box_caption": "%s %.3f" % (f'Flower_{i}', pred["scores"][i]),
                    "scores": {"class_score": pred["scores"][i]},
                    "domain": "pixel",
                }
            )
            class_labels[i] = f'Flower_{i}'
            avg_conf_per_class[pred_class] = avg_conf_per_class[pred_class] + pred["scores"][i]
            if pred_class in counts:
                counts[pred_class] = counts[pred_class] + 1
            else:
                counts[pred_class] = 1

        for pred_class in counts.keys():
            avg_conf_per_class[pred_class] = avg_conf_per_class[pred_class] / counts[pred_class]

        boxes = {
            "predictions": {
                "box_data": boxes_data,
                "class_labels": class_labels,
            }
        }

    masks = {}
    # pixles_per_class = [0 for _ in self.stuff_class_names]
    # Process semantic segmentation predictions
    if pred.get("sem_mask") is not None:
        masks["semantic_mask"] = {
            "mask_data": pred["sem_mask"],
            "class_labels": {1: "Flower"},
        }
        classes = ['Flower']


    # Process instance segmentation detections
    if pred.get("pred_masks") is not None:
        class_count = {}
        num_pred = min(15, len(pred["pred_masks"]))  # Hardcoded to max 15 masks for better UI
        # Sort masks by score
        pred["pred_masks"] = pred["pred_masks"][np.argsort(pred["scores"])[::-1]]
        masks = np.zeros_like(pred["pred_masks"][0]).astype(np.int8)
        class_labels = {}
        for i in range(num_pred):
            pred_class = int(pred["classes"][i])
            if pred_class in class_count:
                class_count[pred_class] = class_count[pred_class] + 1
            else:
                class_count[pred_class] = 0

            mask_title = (
                "Flower"
            )
            mask_title = f"{mask_title}_{i}"
            class_labels[i] = mask_title
            masks += (pred["pred_masks"][i]).astype(np.int8) * (i + 1)

    table_row = [
        id,
        wandb.Image(img, boxes=boxes, masks={"predictions":{"mask_data":masks, "class_labels":class_labels}}),#, classes=classes),
        wandb.Image(detectron_img)
    ]


    return table_row


def log_segmentation_results(predictor, test_data_loader, num_samples, image_type="flower_images"):
    table_data = []

    for i, data in enumerate(test_data_loader):
        if i >= num_samples:
            break

        img = data[image_type].numpy().astype("float32")

        # img = cv2.resize(img, (1333, 889))
        outputs = predictor(img)
        out_cpu = outputs["instances"].to("cpu")
        if len(out_cpu) > 0:
            try:
                v = Visualizer(img[:, :, ::-1],
                metadata=None,
                scale=0.5,
                instance_mode=ColorMode.IMAGE)
                out = v.draw_instance_predictions(out_cpu)
                detectron_img = out.get_image()
                table_data.append(_plot_table_row(_parse_prediction(outputs), img, i, detectron_img))
            except:
                print("Detectron2 Vis Error")
                table_data.append(_plot_table_row(_parse_prediction(outputs), img, i, img))

    columns = ["ID", "Visualization", "Detectron2 Visualization"]
    results_table = wandb.Table(data=table_data, columns=columns)
    wandb.log({"segmentation_results": results_table})
