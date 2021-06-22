from absl import flags
from tensorflow.keras.layers import Dropout, Conv1D, Input, Lambda
from tensorflow.keras import Model
import tensorflow as tf

from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import f1_score, iou_score

import myFlags
FLAGS = flags.FLAGS

def build_seg_model(input_shape=(224, 224, 81)):
    model = Unet('resnet18', encoder_weights=None, classes=2,
                 activation='softmax', input_shape=input_shape)
    model.compile('Adam', loss=bce_jaccard_loss,
                  metrics=[iou_score, f1_score])
    return model

def build_pixel_mlp_class_model(nb_classes, input_shape=(1024,81), loss_weight=0.):
    inputs = Input(input_shape)
    #x = Lambda(lambda x: K.squeeze(x, axis=-1))(inputs)

    # MLP for all pixels
    x = Conv1D(FLAGS.mlp_nodes, 1, strides=1)(inputs)
    x = Dropout(0.5)(x)
    x = Conv1D(FLAGS.mlp_nodes, 1, strides=1, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Conv1D(FLAGS.penultimate_nodes, 1, strides=1,
           activity_regularizer=VarRegularizer(loss_weight), name="penultimate")(x)
    x = Dropout(rate=0.25)(x)
    x = Conv1D(nb_classes, 1, strides=1, activation="softmax", name="predictions")(x)
    preds = Lambda(lambda x: tf.reduce_mean(x, axis=1))(x)

    model = Model(inputs=inputs, outputs=preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer="Adam", metrics=['accuracy'])
    return model

@tf.keras.utils.register_keras_serializable(package='Custom', name='var')
class VarRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, loss_weight=0.):
        self.loss_weight =  loss_weight
    
    def __call__(self, x):
        return self.loss_weight * tf.math.reduce_sum(tf.math.reduce_variance(x, axis=0))
    
    def get_config(self):
        return {'var': float(self.loss_weight)}