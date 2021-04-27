from tensorflow import keras
import tensorflow as tf
from absl import logging
import numpy as np
import cv2

def draw_grid(GRID = 13):
    img = cv2.imread("output.jpg")
    height, width, channels = img.shape
    GRID_SIZE = width/GRID

    verticalPositions = np.arange(GRID_SIZE, width, GRID_SIZE)
    for x in verticalPositions:
        x = int(np.rint(x)) 
        cv2.line(img, (x, 0), (x, height), (0, 0, 0), 1, 1)

    GRID_SIZE = height/GRID
    horizontalPositions = np.arange(GRID_SIZE, height, GRID_SIZE)
    for y in horizontalPositions:
        y = int(np.rint(y)) 
        cv2.line(img, (0, y), (width, y), (0, 0, 0), 1, 1)

    cv2.imwrite(f"output_grid_{GRID}.jpg", img)

def generate_gradcam_heatmap(model, img, class_names):
    print("Generating Grad Cam heatmaps")
    grad_negated_version = [
        False,
        True
    ]
    last_conv_layer_name = "add_22"
    classifier_layer_names = [
        [
            "yolo_conv_0",
            "yolo_output_0",
            "yolo_boxes_0"
        ]
        ,
        [
            "yolo_conv_0",
            "yolo_conv_1",
            "yolo_output_1",
            "yolo_boxes_1",
        ],
        [
            "yolo_conv_0",
            "yolo_conv_1",
            "yolo_conv_2",
            "yolo_output_2",
            "yolo_boxes_2",
        ]
    ]
    generate_for_classes=[0,1,2,3,7,9,15,16,19,26]
    generate_for_classes=[15, 0, 26, 2, 75, 58, 13]
    generate_for_classes=[0, 73, 75, 71, 70]
    generate_for_classes=[75, 0, 2]
    generate_for_classes = [5,0,2,7]
    # generate_for_classes = [8]
    # generate_for_classes = [15, 16]
    generate_for_classes = [0, 2]
    generate_for_classes = [8]
    # for classificationLayerSize in classifier_layer_names:
    prefix = "grad"
    for negatedGradParameter in grad_negated_version:
        if negatedGradParameter:
            prefix = "grad_negated"
            print("--Negated Grad-CAM")
        else:
            print("--Grad-CAM")
        for class_index in generate_for_classes:
            print("Class:", class_names[class_index])
            heatmaps = []
            normalizedHeatmaps = []
            maxVals = []
            for i,classificationLayerSize in enumerate(classifier_layer_names):
                heatmap = _make_heatmap(img, model, last_conv_layer_name, classificationLayerSize, class_names, class_index, negative_grad=negatedGradParameter)
                heatmaps.append(heatmap)
                maxVals.append(np.max(heatmap))
            maxVal = np.max(maxVals)
            print("Vals:", maxVals)
            print("maxVal:", maxVal)
            for i,heatmap in enumerate(heatmaps):
                gridS = 13
                if i == 1:
                    gridS = 26
                elif i == 2:
                    gridS = 52
                draw_grid(gridS)
                heatmap = _normalize_heatmap(heatmap, maxVal)
                normalizedHeatmaps.append(heatmap)
                _augment_image(heatmap, i, save_path_base=f"{prefix}_{class_names[class_index]}", base_img_path=f"output_grid_{gridS}.jpg")
                _fade_out_augmentation(heatmap, i, save_path_base=f"{prefix}_{class_names[class_index]}_faded", base_img_path=f"output_grid_{gridS}.jpg")
            # combined_heatmap = np.prod(heatmaps, axis=0)
            # combined_heatmap = np.clip(combined_heatmap, 0, 1)
            combined_heatmap = np.maximum.reduce(normalizedHeatmaps)
            _augment_image(combined_heatmap, "combined", f"{prefix}_{class_names[class_index]}")
            _fade_out_augmentation(combined_heatmap, "combined", f"{prefix}_{class_names[class_index]}_faded")
    print("All Grad-Cam visualizations generated")


def _make_heatmap(img, model, last_conv_layer_name, classifier_layer_names, class_names, pred_index=0, negative_grad=True):
    yolo_backbone = model.get_layer("yolo_darknet")
    # yolo_backbone.summary()
    last_conv_layer = yolo_backbone.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(yolo_backbone.inputs, last_conv_layer.output)


    # x_36, x_61, x = yolo_backbone.output
    last_conv_layer_61 = yolo_backbone.get_layer("add_18")
    x_61 = keras.Input(shape=last_conv_layer_61.output.shape[1:])
    
    last_conv_layer_36 = yolo_backbone.get_layer("add_10")
    x_36 = keras.Input(shape=last_conv_layer_36.output.shape[1:])

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        if layer_name == "yolo_conv_1":
            x = model.get_layer(layer_name)((x, x_61))
        elif layer_name == "yolo_conv_2":
             x = model.get_layer(layer_name)((x, x_36))
        else:
            x = model.get_layer(layer_name)(x)
    c_input = classifier_input
    classifier_suffix = "1"
    if "yolo_conv_2" in classifier_layer_names:
        c_input = (classifier_input, x_61, x_36)
        classifier_suffix = "3"
    elif "yolo_conv_1" in classifier_layer_names:
        c_input = (classifier_input, x_61)
        classifier_suffix = "2"
    classifier_model = keras.Model(c_input, x)
    tf.keras.utils.plot_model(classifier_model, to_file=f'model_classifier_td_{classifier_suffix}.png', show_shapes=False, expand_nested=False)

    # nms_layer = model.get_layer("yolo_nms")
    # nms_layer_model = keras.Model(nms_layer.inputs, nms_layer.output)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        # last_conv_layer_output = last_conv_layer_model(img)
        x_36, x_64, last_conv_layer_output = yolo_backbone(img)
        # Compute class predictions
        # preds = classifier_model(last_conv_layer_output)
        # print(preds)
        classifier_input = last_conv_layer_output
        if "yolo_conv_2" in classifier_layer_names:
            classifier_input = (classifier_input, x_64, x_36)
        elif "yolo_conv_1" in classifier_layer_names:
            classifier_input = (classifier_input, x_64)

        tape.watch(classifier_input)

        bbox, objectness, class_probs, pred_box = classifier_model(classifier_input)
        # boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4))
        boxes = tf.reshape(bbox, (tf.shape(bbox)[0], -1, tf.shape(bbox)[-1]))
        new_objectness = tf.reshape(objectness, (tf.shape(objectness)[0], -1, tf.shape(objectness)[-1]))
        new_class_probs = tf.reshape(class_probs, (tf.shape(class_probs)[0], -1, tf.shape(class_probs)[-1]))
        # print("boxes:", boxes.shape)
        # print("new_objectness:", new_objectness.shape)
        # print("new_class_probs:", new_class_probs.shape)
        # print(preds)
        top_class_channel = new_class_probs[0, :, pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    # grads = grads * -1
    if negative_grad:
        grads = np.negative(grads)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # maxi = np.max(heatmap)
    # print(maxi)
    # heatmap = np.maximum(heatmap, 0) / maxi
    heatmap = np.maximum(heatmap, 0)
    return heatmap

def _normalize_heatmap(heatmap, maximalVal):
    heatmap = heatmap / maximalVal
    # heatmap = np.clip(heatmap*10, 0, 1)
    return heatmap

def _augment_image(heatmap, save_path_appendix="", save_path_base="gradcam_result", alpha = 0.9, base_img_path="./output.jpg"):
    # We use cv2 to load the original image
    img_path = base_img_path
    img = cv2.imread(img_path)

    # We resize the heatmap to have the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.clip(heatmap, 0, 1)

    # We convert the heatmap to RGB
    heatmap = np.uint8(255 * heatmap)

    # We apply the heatmap to the original image
    colormap = cv2.COLORMAP_JET
    # colormap = cv2.COLORMAP_PARULA
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # 0.4 here is a heatmap intensity factor
    superimposed_img = heatmap + alpha * img
    # heatmap = heatmap.reshape(*heatmap.shape, 1)
    # superimposed_img = img * heatmap

    # Save the image to disk
    save_path = f'./gradcam/{save_path_base}_{save_path_appendix}.jpg'
    cv2.imwrite(save_path, superimposed_img)

def _fade_out_augmentation(heatmap, save_path_appendix="", save_path_base="gradcam_result", alpha = 0.9, base_img_path="./output.jpg"):
    # We use cv2 to load the original image
    img_path = base_img_path
    img = cv2.imread(img_path)

    # We resize the heatmap to have the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.clip(heatmap, 0, 1)

    heatmap = heatmap.reshape(*heatmap.shape, 1)
    superimposed_img = img * heatmap

    # Save the image to disk
    save_path = f'./gradcam/{save_path_base}_{save_path_appendix}.jpg'
    cv2.imwrite(save_path, superimposed_img)