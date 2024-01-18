import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import imgaug
# Root directory of the project
ROOT_DIR = os.path.abspath("")
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import model as modellib, utils
 
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
 
#last path
LAST_PATH = os.path.join(ROOT_DIR, "logs\\granulo20210215T1719\\mask_rcnn_granulo_0044.h5")
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


 
############################################################
#  Configurations
############################################################
 
 
class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Granulo_MM"
 
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 2  #  grain + bckgrnd
 
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 250
    
    # Skip detections with < x% confidence
    DETECTION_MIN_CONFIDENCE = 0
    
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 768 #default 800 a 1024
    
    DETECTION_MAX_INSTANCES = 1200  #Max number of final detections default 100
    RPN_NMS_THRESHOLD = 0.5 # Non-max suppression threshold to filter RPN proposals. default 0.7
    POST_NMS_ROIS_INFERENCE = 1200 # ROIs kept after non-maximum suppression (training and inference) default 1000
    DETECTION_NMS_THRESHOLD = 0.3 # Non-maximum suppression threshold for detection default 0.3
    PRE_NMS_LIMIT = 6000 # ROIs kept after tf.nn.top_k and before non-maximum suppression default 6000
    RPN_TRAIN_ANCHORS_PER_IMAGE = 1000
    TRAIN_ROIS_PER_IMAGE = 400 #default 200
    
    
    RPN_ANCHOR_SCALES = (16,32, 64, 128,256)

    
    MAX_GT_INSTANCES=1500
    
    LEARNING_RATE=0.001
    
    IMAGE_RESIZE_MODE="crop"
    
    
    
 
 
############################################################
#  Dataset
############################################################
 
class CustomDataset(utils.Dataset):
 
    def load_custom(self, dataset_dir, subset):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "grain")
 
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
 
        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations1 = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        # print(annotations1)
        annotations = list(annotations1.values())  # don't need the dict keys
 
        # Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
 
        # Add images
        for a in annotations:
            
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes']['objects'] for s in a['regions']]
       
            name_dict = {"grain": 1}
            # key = tuple(name_dict)
            num_ids = [name_dict[a] for a in objects]
 
           
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
 
            self.add_image(
                "object", 
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)
 
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        #delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
 
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            try:
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                mask[rr, cc, i] = 1
            except:
                continue
 
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids
 
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
 
 
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()
 
    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()
    
    
    augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0)),
                    imgaug.augmenters.color.AddToBrightness(add=(-30, 30)),
                    imgaug.augmenters.color.AddToHueAndSaturation()
                ])
                
    
    
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE*5,
                epochs=5,
                layers='rpn')
    """              
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE*3,
                epochs=10,
                layers='heads')
             
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=100,
            layers='heads',
            augmentation = augmentation)
    """
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=15,
                layers='all')
                
         
    """
    #heads
    
                
    
    
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/2,
                epochs=100,
                layers='heads')
                
    print("Training rpn")
                
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/3,
                epochs=110,
                layers='rpn')  
    """            
    """
    #pyramid
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=400,
                layers='5+')
                
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=500,
                layers='4+')
                
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=600,
                layers='3+')
                
    #refining  
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/10,
                epochs=1000,
                layers='all')         
    
    """
############################################################
#  Training
############################################################
 
if __name__ == '__main__':
    import argparse
 
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()
 
    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"
 
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
 
    # Configurations
    config = CustomConfig()
    config.display()
 
    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = LAST_PATH
    else:
        weights_path = args.weights
 
    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
 
    train(model)