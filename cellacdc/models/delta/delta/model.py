"""
Module for model definitions and loss/metrics functions definitions

@author: jblugagne
"""
from typing import Union, List, Tuple, Callable, Dict

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    UpSampling2D,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam


#%% Losses/metrics


def pixelwise_weighted_binary_crossentropy_seg(
    y_true: tf.Tensor, y_pred: tf.Tensor
) -> tf.Tensor:
    """
    Pixel-wise weighted binary cross-entropy loss.
    The code is adapted from the Keras TF backend.
    (see their github)

    Parameters
    ----------
    y_true : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    y_pred : Tensor
        Predicted segmentation masks.

    Returns
    -------
    Tensor
        Pixel-wise weight binary cross-entropy between inputs.

    """
    try:
        # The weights are passed as part of the y_true tensor:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass

    # Make background weights be equal to the model's prediction
    bool_bkgd = weight == 0 / 255
    weight = tf.where(bool_bkgd, y_pred, weight)

    epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    cond = y_pred >= zeros
    relu_logits = math_ops.select(cond, y_pred, zeros)
    neg_abs_logits = math_ops.select(cond, -y_pred, y_pred)
    entropy = math_ops.add(
        relu_logits - y_pred * seg,
        math_ops.log1p(math_ops.exp(neg_abs_logits)),
        name=None,
    )

    loss = K.mean(math_ops.multiply(weight, entropy), axis=-1)

    loss = tf.scalar_mul(
        10 ** 6, tf.scalar_mul(1 / tf.math.sqrt(tf.math.reduce_sum(weight)), loss)
    )

    return loss


def pixelwise_weighted_binary_crossentropy_track(
    y_true: tf.Tensor, y_pred: tf.Tensor
) -> tf.Tensor:
    """
    Pixel-wise weighted binary cross-entropy loss.
    The code is adapted from the Keras TF backend.
    (see their github)

    Parameters
    ----------
    y_true : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    y_pred : Tensor
        Predicted segmentation masks.

    Returns
    -------
    Tensor
        Pixel-wise weight binary cross-entropy between inputs.

    """
    try:
        # The weights are passed as part of the y_true tensor:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass

    epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    cond = y_pred >= zeros
    relu_logits = math_ops.select(cond, y_pred, zeros)
    neg_abs_logits = math_ops.select(cond, -y_pred, y_pred)
    entropy = math_ops.add(
        relu_logits - y_pred * seg,
        math_ops.log1p(math_ops.exp(neg_abs_logits)),
        name=None,
    )

    loss = K.mean(math_ops.multiply(weight, entropy), axis=-1)

    loss = tf.scalar_mul(
        10 ** 6, tf.scalar_mul(1 / tf.math.sqrt(tf.math.reduce_sum(weight)), loss)
    )

    return loss


def class_weighted_categorical_crossentropy(
    class_weights: Union[List[float], Tuple[float, ...]]
) -> Callable[[tf.Tensor, tf.Tensor], float]:
    """
    Generate class-weighted categorical cross-entropy loss function.
    The code is adapted from the Keras TF backend.
    (see their github)

    Parameters
    ----------
    class_weights : tuple/list of floats
        Weights for each class/category.

    Returns
    -------
    function.
        Class-weighted categorical cross-entropy loss function.

    """

    def loss_function(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:

        # scale preds so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, -1, True)
        # manual computation of crossentropy
        epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Multiply each class by its weight:
        classes_list = tf.unstack(y_true * tf.math.log(y_pred), axis=-1)
        for i in range(len(classes_list)):
            classes_list[i] = tf.scalar_mul(class_weights[i], classes_list[i])

        # Return weighted sum:
        return -tf.reduce_sum(tf.stack(classes_list, axis=-1), -1)

    return loss_function


def unstack_acc(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Unstacks the mask from the weights in the output tensor for
    segmentation and computes binary accuracy

    Parameters
    ----------
    y_true : Tensor
        Stack of groundtruth segmentation masks + weight maps.
    y_pred : Tensor
        Predicted segmentation masks.

    Returns
    -------
    Tensor
        Binary prediction accuracy.

    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass

    return keras.metrics.binary_accuracy(seg, y_pred)


#%% Models
# Generic unet declaration:
def unet(
    input_size: Tuple[int, int, int] = (256, 32, 1),
    final_activation: str = "sigmoid",
    output_classes: int = 1,
    dropout: float = 0,
    levels: int = 5,
) -> Model:
    """
    Generic U-Net declaration.

    Parameters
    ----------
    input_size : tuple of 3 ints, optional
        Dimensions of the input tensor, excluding batch size.
        The default is (256,32,1).
    final_activation : string or function, optional
        Activation function for the final 2D convolutional layer. see
        keras.activations
        The default is 'sigmoid'.
    output_classes : int, optional
        Number of output classes, ie dimensionality of the output space of the
        last 2D convolutional layer.
        The default is 1.
    dropout : float, optional
        Dropout layer rate in the contracting & expanding blocks. Valid range
        is [0,1). If 0, no dropout layer is added.
        The default is 0.
    levels : int, optional
        Number of levels of the U-Net, ie number of successive contraction then
        expansion blocks are combined together.
        The default is 5.

    Returns
    -------
    model : Model
        Defined U-Net model (not compiled yet).

    """

    # Default conv2d parameters:
    conv2d_parameters = {
        "activation": "relu",
        "padding": "same",
        "kernel_initializer": "he_normal",
    }

    # Inputs layer:
    inputs = Input(input_size, name="true_input")

    # First level input convolutional layers:
    filters = 64
    conv = Conv2D(filters, 3, **conv2d_parameters, name="Level0_Conv2D_1")(inputs)
    conv = Conv2D(filters, 3, **conv2d_parameters, name="Level0_Conv2D_2")(conv)

    # Generate contracting path:
    level = 0
    contracting_outputs = [conv]
    for level in range(1, levels):
        filters *= 2
        contracting_outputs.append(
            contracting_block(
                contracting_outputs[-1],
                filters,
                conv2d_parameters,
                dropout=dropout,
                name=f"Level{level}_Contracting",
            )
        )

    # Generate expanding path:
    expanding_output = contracting_outputs.pop()
    while level > 0:
        level -= 1
        filters = int(filters / 2)
        expanding_output = expanding_block(
            expanding_output,
            contracting_outputs.pop(),
            filters,
            conv2d_parameters,
            dropout=dropout,
            name=f"Level{level}_Expanding",
        )

    # Final output layer:
    output = Conv2D(output_classes, 1, activation=final_activation, name="true_output")(
        expanding_output
    )

    model = Model(inputs=inputs, outputs=output)

    return model


def contracting_block(
    input_layer: tf.Tensor,
    filters: int,
    conv2d_parameters: Dict,
    dropout: float = 0,
    name: str = "Contracting",
) -> tf.Tensor:
    """
    A block of layers for 1 contracting level of the U-Net

    Parameters
    ----------
    input_layer : tf.Tensor
        The convolutional layer that is the output of the upper level's
        contracting block.
    filters : int
        filters input for the Conv2D layers of the block.
    conv2d_parameters : dict()
        kwargs for the Conv2D layers of the block.
    dropout : float, optional
        Dropout layer rate in the block. Valid range is [0,1). If 0, no dropout
        layer is added.
        The default is 0
    name : str, optional
        Name prefix for the layers in this block. The default is "Contracting".

    Returns
    -------
    conv2 : tf.Tensor
        Output of this level's contracting block.

    """

    # Pooling layer: (sample 'images' down by factor 2)
    pool = MaxPooling2D(pool_size=(2, 2), name=name + "_MaxPooling2D")(input_layer)

    # First convolution layer:
    conv1 = Conv2D(filters, 3, **conv2d_parameters, name=name + "_Conv2D_1")(pool)

    # Second convolution layer:
    conv2 = Conv2D(filters, 3, **conv2d_parameters, name=name + "_Conv2D_2")(conv1)

    # Add dropout if necessary, otherwise return:
    if dropout == 0:
        return conv2
    else:
        drop = Dropout(dropout, name=name + "_Dropout")(conv2)
        return drop


def expanding_block(
    input_layer: tf.Tensor,
    skip_layer: tf.Tensor,
    filters: int,
    conv2d_parameters: Dict,
    dropout: float = 0,
    name: str = "Expanding",
) -> tf.Tensor:
    """
    A block of layers for 1 expanding level of the U-Net

    Parameters
    ----------
    input_layer : tf.Tensor
        The convolutional layer that is the output of the lower level's
        expanding block
    skip_layer : tf.Tensor
        The convolutional layer that is the output of this level's
        contracting block
    filters : int
        filters input for the Conv2D layers of the block.
    conv2d_parameters : dict()
        kwargs for the Conv2D layers of the block.
    dropout : float, optional
        Dropout layer rate in the block. Valid range is [0,1). If 0, no dropout
        layer is added.
        The default is 0
    name : str, optional
        Name prefix for the layers in this block. The default is "Expanding".

    Returns
    -------
    conv3 : tf.Tensor
        Output of this level's expanding block.

    """

    # Up-sampling:
    up = UpSampling2D(size=(2, 2), name=name + "_UpSampling2D")(input_layer)
    conv1 = Conv2D(filters, 2, **conv2d_parameters, name=name + "_Conv2D_1")(up)

    # Merge with skip connection layer:
    merge = Concatenate(axis=3, name=name + "_Concatenate")([skip_layer, conv1])

    # Convolution layers:
    conv2 = Conv2D(filters, 3, **conv2d_parameters, name=name + "_Conv2D_2")(merge)
    conv3 = Conv2D(filters, 3, **conv2d_parameters, name=name + "_Conv2D_3")(conv2)

    # Add dropout if necessary, otherwise return:
    if dropout == 0:
        return conv3
    else:
        drop = Dropout(dropout, name=name + "_Dropout")(conv3)
        return drop


# Use the following model for segmentation:
def unet_seg(
    pretrained_weights: str = None,
    input_size: Tuple[int, int, int] = (256, 32, 1),
    levels: int = 5,
) -> Model:
    """
    Cell segmentation U-Net definition function.

    Parameters
    ----------
    pretrained_weights : hdf5 file, optional
        Model will load weights from hdf5 and start training.
        The default is None
    input_size : tuple of 3 ints, optional
        Dimensions of the input tensor, without batch size.
        The default is (256,32,1).
    levels : int, optional
        Number of levels of the U-Net, ie number of successive contraction then
        expansion blocks are combined together.
        The default is 5.

    Returns
    -------
    model : Model
        Segmentation U-Net (compiled).

    """

    model = unet(
        input_size=input_size,
        final_activation="sigmoid",
        output_classes=1,
        levels=levels,
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=pixelwise_weighted_binary_crossentropy_seg,
        metrics=[unstack_acc],
    )

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# Use the following model for tracking and lineage reconstruction:
def unet_track(
    pretrained_weights: str = None,
    input_size: Tuple[int, int, int] = (256, 32, 4),
    levels: int = 5,
) -> Model:
    """
    Tracking U-Net definition function.

    Parameters
    ----------
    pretrained_weights : hdf5 file, optional
        Model will load weights from hdf5 and start training.
        The default is None
    input_size : tuple of 3 ints, optional
        Dimensions of the input tensor, without batch size.
        The default is (256,32,4).
    levels : int, optional
        Number of levels of the U-Net, ie number of successive contraction then
        expansion blocks are combined together.
        The default is 5.

    Returns
    -------
    model : Model
        Tracking U-Net (compiled).

    """

    model = unet(
        input_size=input_size,
        final_activation="sigmoid",
        output_classes=1,
        levels=levels,
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=pixelwise_weighted_binary_crossentropy_track,
        metrics=[unstack_acc],
    )

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


# Use the following model for segmentation:
def unet_rois(
    input_size: Tuple[int, int, int] = (512, 512, 1), levels: int = 5
) -> Model:
    """
    ROIs segmentation U-Net.

    Parameters
    ----------
    input_size : tuple of 3 ints, optional
        Dimensions of the input tensor, without batch size.
        The default is (512,512,1).
    levels : int, optional
        Number of levels of the U-Net, ie number of successive contraction then
        expansion blocks are combined together.
        The default is 5.

    Returns
    -------
    model : Model
        ROIs ID U-Net (compiled).

    """

    model = unet(
        input_size=input_size,
        final_activation="sigmoid",
        output_classes=1,
        levels=levels,
    )

    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy")

    return model
