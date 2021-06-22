import os
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from absl import app
from absl import flags
from albumentations import (
    Compose, HorizontalFlip, RandomBrightness,RandomContrast,
    ShiftScaleRotate, ToFloat, VerticalFlip)

from models import build_seg_model, build_pixel_mlp_class_model
from utils import reset_tf, set_seed
from VegetableSequence import VegetableDataset, VegetableSequence
import myFlags

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
FLAGS = flags.FLAGS

def main(argv):
    reset_tf(FLAGS.device)
    set_seed()

    isSeg = FLAGS.isSeg # train segmentation model

    # data structure
    ds_info = VegetableDataset(FLAGS.data_path)
    dim = ds_info.hsi_dims
    input_shape = (224, 224, dim) if isSeg else (FLAGS.nb_pixels, dim)

    # Experiment name
    experiment_title = "HSSD" 
    experiment_title += '-seg' if isSeg else '-pix_class'
    experiment_title += '-%d' % time.time()
    logdir = os.path.join(FLAGS.log_root, experiment_title)
    os.mkdir(logdir)
    print("logdir: ", logdir)

    # augmentation
    if isSeg:
        AUGMENTATIONS_TRAIN = Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.2),
            RandomContrast(limit=0.001, p=0.5),
            RandomBrightness(limit=0.001, p=0.5),
            ShiftScaleRotate(
                shift_limit=0.3, scale_limit=0.9,
                rotate_limit=30, border_mode=4, p=0.8),# cv2.BORDER_REFLECT_101
            ToFloat(max_value=1024)
        ])
    else:
        AUGMENTATIONS_TRAIN = Compose([
            RandomContrast(limit=0.001, p=0.5),
            RandomBrightness(limit=0.001, p=0.5),
            ToFloat(max_value=1024)
        ])
    AUGMENTATIONS_TEST = AUGMENTATIONS_TRAIN

    # loading dataset
    train_gen = VegetableSequence(FLAGS.batch_size, instance_ids=[1, 2, 3],
                        sample_ids=[1,2], dataset=ds_info, isSeg=isSeg,
                        nb_pixels=FLAGS.nb_pixels,augmentations=AUGMENTATIONS_TRAIN)
    valid_gen = VegetableSequence(FLAGS.batch_size, instance_ids=[4],
                        sample_ids=[1,2], dataset=ds_info, isSeg=isSeg,
                        nb_pixels=FLAGS.nb_pixels,augmentations=AUGMENTATIONS_TEST,
                        random_state=2021)

    # building a model
    if isSeg:
        model = build_seg_model(input_shape=input_shape)
    else:
        model = build_pixel_mlp_class_model(
                nb_classes=ds_info.object_categories, input_shape=input_shape,
                loss_weight=FLAGS.loss_weight)

    # callbacks
    checkpoint = ModelCheckpoint(logdir + "/best.weights.hdf5", monitor='val_loss',
                                 save_best_only=True, save_weights_only=True,
                                 mode='auto', save_freq="epoch")
    early_stopping = EarlyStopping(monitor="val_loss", patience=FLAGS.patience)
    callbacks = [checkpoint, early_stopping]
    
    model.fit(train_gen, epochs=FLAGS.epochs,validation_data=valid_gen,
              validation_steps=len(valid_gen), callbacks=callbacks)

if __name__ == "__main__":
    app.run(main)
