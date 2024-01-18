# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:30:41 2021

@author: Utilisateur
"""

import os
import sys
import matplotlib.pyplot as plt
import skimage
import numpy as np
 
# Root directory of the project
ROOT_DIR = os.path.abspath("")
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils
import grain
 
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
    
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
# Path to trained weights
# GRAIN_WEIGHTS_PATH = "C:/Users/Utilisateur/Desktop/Granulo_project/Grain_RCNN_V3/logs/granulo_mm20210822T1206/mask_rcnn_granulo_mm_0181.h5"
# GRAIN_WEIGHTS_PATH = "C:/Users/Utilisateur/Desktop/Granulo_project/Grain_RCNN_V3/logs/granulo_mm20210824T0837/mask_rcnn_granulo_mm_0002.h5"
GRAIN_WEIGHTS_PATH = "C:/Users/Utilisateur/Desktop/Granulo_project/Grain_RCNN_V3/logs/granulo_mm20210824T1043/mask_rcnn_granulo_mm_0004.h5"




config = grain.CustomConfig()
CUSTOM_DIR = os.path.join(ROOT_DIR, "dataset")
 
# changes for inferencing.
class InferenceConfig(config.__class__):
    MAX_GT_INSTANCES = 1000
    POST_NMS_ROIS_INFERENCE = 5000
    PRE_NMS_LIMIT = 15000
    IMAGE_MAX_DIM = 768
    IMAGE_RESIZE_MODE="square"
    RPN_NMS_THRESHOLD = 0.7
    DETECTION_MAX_INSTANCES=1000
    #RPN_ANCHOR_SCALES = (32, 64, 128, 256, 16)
    DETECTION_NMS_THRESHOLD = 0.3
    DETECTION_MIN_CONFIDENCE=0
    
    
config = InferenceConfig()
config.display()
 
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir = MODEL_DIR,config=config)

# Load weights
weights_path = GRAIN_WEIGHTS_PATH
model.load_weights(weights_path, by_name=True)

#load image
im_path = "C:/Users/Utilisateur/Desktop/Granulo_project/image_from_sedinet/Cal_08.tif"

#im_path = "C:/Users/Utilisateur/Desktop/Granulo_project/Prerouge/cut_grille_com_epch200/decoupe_3.tif"
#im_path = "C:/Users/Utilisateur/Desktop/Granulo_project/Prerouge/auto_programa/tmp7y2zvsfi/cut_2_28.tif"
#im_path="C:/Users/Utilisateur/Desktop/Granulo_project/Photos_molliere/IMG_20210603_095659.jpg"




#512pxl
# im_path = "C:/Users/Utilisateur/Desktop/Granulo_project/Prerouge/1m5/1.tif" 
#im_path = "C:/Users/Utilisateur/Desktop/Granulo_project/Grain_RCNN_V3/dataset/val/im_1.JPG"
#1000
# im_path = "C:/Users/Utilisateur/Desktop/Granulo_project/bernard/Capture.jpg"

#2000+
# im_path="C:/Users/Utilisateur/Desktop/Granulo_project/Photos_molliere/cut1.jpg"

#4000+
# im_path="C:/Users/Utilisateur/Desktop/Granulo_project/Photos_molliere/IMG_20210603_184436.jpg"


image = skimage.io.imread(im_path)

# original image
plt.figure(figsize=(12,10))
skimage.io.imshow(image)

# If grayscale. Convert to RGB for consistency.
if image.ndim != 3:
    image = skimage.color.gray2rgb(image)
# If has an alpha channel, remove it for consistency
if image.shape[-1] == 4:
    image = image[..., :3]
    
# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
class_names = ['BG', 'grain']
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, None, show_bbox=False)

"""

pillar = model.keras_model.get_layer("ROI").output 
# TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
if nms_node is None:
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
if nms_node is None: #TF 1.9-1.10
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")

rpn = model.run_graph([image], [
    ("rpn_class", model.keras_model.get_layer("rpn_class").output),
    ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
    ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
    ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
    ("post_nms_anchor_ix", nms_node),
    ("proposals", model.keras_model.get_layer("ROI").output),
])

# Show top anchors by score (before refinement)
limit = 100
sorted_anchor_ids = np.argsort(rpn['rpn_class'][:,:,1].flatten())[::-1]
visualize.draw_boxes(image, boxes=model.anchors[sorted_anchor_ids[:limit]], ax=get_ax())

# Show top anchors with refinement. Then with clipping to image boundaries
limit = 50
ax = get_ax(1, 2)
pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
visualize.draw_boxes(image, boxes=pre_nms_anchors[:limit],
                     refined_boxes=refined_anchors[:limit], ax=ax[0])
visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[:limit], ax=ax[1])

# Show refined anchors after non-max suppression
limit = 500
ixs = rpn["post_nms_anchor_ix"][:limit]
visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[ixs], ax=get_ax())

# Show final proposals
# These are the same as the previous step (refined anchors 
# after NMS) but with coordinates normalized to [0, 1] range.
limit = 100
# Convert back to image coordinates for display
h, w = config.IMAGE_SHAPE[:2]
proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
visualize.draw_boxes(image, refined_boxes=proposals, ax=get_ax())

"""

