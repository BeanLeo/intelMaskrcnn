import os
import random
import math
import tensorflow as tf
import argparse
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utils,config 
import time
from mrcnn import visualize
ROOT_DIR = r'./'
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
inferconfig = config.Config()
inferconfig.NUM_CLASSES = 81
#inferconfig.IMAGES_PER_GPU = 1
inferconfig.BATCH_SIZE = 1
def unmold_detections(detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(masks.shape[1:3] + (0,))

        return boxes, class_ids, scores, full_masks
def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.
    
    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])
def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta
def mold_image(images):
    """Expects an RGB image (or array of images) and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - inferconfig.MEAN_PIXEL
def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)
    file_names = next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(os.path.join(IMAGE_DIR,'test.jpg')) #random.choice(file_names)))
    # We can verify that we can access the list of operations in the graph
    molded_image,inferwindow,scale,padding,crop = utils.resize_image(
        image,
        min_dim = inferconfig.IMAGE_MIN_DIM,
        min_scale = inferconfig.IMAGE_MIN_SCALE,
        max_dim = inferconfig.IMAGE_MAX_DIM,
        mode = inferconfig.IMAGE_RESIZE_MODE)
    molded_image = mold_image(molded_image)
    print("Moded image shape is : ", molded_image.shape)
    image_meta = compose_image_meta(0,image.shape, molded_image.shape,inferwindow, scale,
        np.zeros([inferconfig.NUM_CLASSES],dtype=np.int32))
    #image =  image[np.newaxis,:]
    #anchors = anchors[np.newaxis, :]
    image_meta = image_meta.reshape(1,-1)

    backbone_shapes = compute_backbone_shapes(inferconfig, molded_image.shape)
    imageshapeinfer = molded_image.shape
    molded_image = molded_image[np.newaxis, :]
    #print("Backbone shape is : ", backbone_shapes)
    anchors = utils.generate_pyramid_anchors(inferconfig.RPN_ANCHOR_SCALES,
                                             inferconfig.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             inferconfig.BACKBONE_STRIDES,
                                             inferconfig.RPN_ANCHOR_STRIDE)
    #print("Anchor generate parameter : ",inferconfig.RPN_ANCHOR_SCALES)
    #print("Anchor generate parameter : ",inferconfig.RPN_ANCHOR_RATIOS)
    #print("Anchor generate paramenter :",backbone_shapes)
    #print("Anchor generate parameter : ",inferconfig.BACKBONE_STRIDES)
    #print("Anchor generate parameter : ",inferconfig.RPN_ANCHOR_STRIDE)
    #print("Original anchor shape is :", anchors.shape)
    anchors = np.broadcast_to(anchors, (inferconfig.BATCH_SIZE,) + anchors.shape)
    anchors = utils.norm_boxes(anchors,imageshapeinfer[:2])
    print("The input anchors shape is : ",anchors.shape)
    print('The input anchors are : \n', anchors)
        #print(image.shape)
    test_list = []
    for count,op in enumerate(graph.get_operations()):
        if "detection" in op.name:
            print(op.name)
            test_list.append(op.name)
    #print(graph.get_operation_by_name('prefix/input_image'))
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
    
    # We access the input and output nodes 
    # x = graph.get_tensor_by_name('prefix/*/inputs_placeholder:0')
    input_image_placeholder = graph.get_tensor_by_name('prefix/input_image:0')
    input_imagemeta_placeholder = graph.get_tensor_by_name('prefix/input_image_meta:0')
    input_anchor_placeholder = graph.get_tensor_by_name('prefix/input_anchors:0')
    output_detections = graph.get_tensor_by_name('prefix/mrcnn_detection/Reshape_1:0')
    output_masks = graph.get_tensor_by_name('prefix/mrcnn_mask/Reshape_1:0')
    test_probe = graph.get_tensor_by_name(test_list[-1]+':0')
    sptest = graph.get_tensor_by_name('prefix/ROI/refined_anchors:0')
    # print(x)
    #y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')
    #    
    ## We launch a Session
    results = []
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants 
        start_time = time.time()
        inter_detections, inter_masks, test, sppb = sess.run([output_detections, output_masks,test_probe,sptest], feed_dict={
            input_image_placeholder:molded_image,
            input_imagemeta_placeholder:image_meta, # < 45
            input_anchor_placeholder:anchors
        })
        end_time = time.time()
        print("Time used for inference on TITAN X GPU is :",end_time-start_time)
        sum_test = np.sum(test,axis = 0)
        #sum_std = np.sum(std)
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
        #print("The anchors are : ",anchors)
        #print("The shape of detections is: ",inter_detections.shape) # [[ False ]] Yay, it works!
        #print("The detections are : \n",inter_detections)
        #print("The shape of mask is: ",inter_masks.shape)
        #print("The shape of sppb is :", sppb.shape)
        #print("Content of sppb is:", sppb)
        #print("The detection is: ", inter_detections)
        #print("The mask is : ",inter_masks)
        #print("The windows is: ",inferwindow[-2:]," of shape : ",inferwindow)
        #print("The image meta is : ",image_meta)
        #print("Test probe "+test_list[-1]+" tensor is: ", test)
        #print("Test probe "+test_list[-1]+" tensor shape is: ", test.shape)
        final_rois, final_class_ids, final_scores, final_masks = \
                     unmold_detections(detections = inter_detections[0], 
                                   mrcnn_mask = inter_masks[0],
                                   original_image_shape = image.shape,
                                   image_shape = molded_image[0].shape,
                                   window = inferwindow)
        results.append({
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        })
        r = results[0]
        class_names = 'cls'*81
        #print('The detection result : \n',r['rois'])
        print("The mask's shape are : \n",r['masks'].shape)
        visualize.save_mask_images(image,r['rois'],r['masks'],r['class_ids'],class_names,r['scores'],is_pb = True)
    
        #print("The class_ids is:",final_class_ids.shape)
        #print("The scores is: ", final_scores.shape)
