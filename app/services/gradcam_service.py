import numpy as np
import tensorflow as tf


def build_gradcam_heatmap(
    processed_image,
    model,
    pred_index,
    backbone_name,
    last_conv_layer_name
):
  
    backbone = model.get_layer(backbone_name)

    last_conv_layer = backbone.get_layer(last_conv_layer_name)

    last_conv_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=last_conv_layer.output
    )

    classifier_input = tf.keras.Input(
        shape=last_conv_layer.output.shape[1:]
    )

    x = classifier_input

    passed_backbone = False

    for layer in model.layers:

        if layer.name == backbone_name:

            passed_backbone = True

            continue

        if passed_backbone:

            x = layer(x)

    classifier_model = tf.keras.models.Model(classifier_input, x)

    with tf.GradientTape() as tape:

        conv_output = last_conv_model(processed_image)

        tape.watch(conv_output)

        predictions = classifier_model(conv_output)

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(
        conv_output * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


def generate_gradcam(
    model,
    processed_image,
    pred_index,
    backbone_name,
    last_conv_layer_name
):
    
    raw_heatmap = (
        build_gradcam_heatmap(
            processed_image=processed_image,
            model=model,
            pred_index=pred_index,
            backbone_name=backbone_name,
            last_conv_layer_name=last_conv_layer_name
        )
    )

    return {
        "raw_heatmap":
            raw_heatmap.astype(
                np.float32
            )
    }