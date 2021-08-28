import os
from absl import app
from absl import flags
import numpy as np
import tqdm

from tensorflow.keras import Model
from albumentations import (
    Compose, HorizontalFlip, RandomBrightness,RandomContrast,
    ShiftScaleRotate, ToFloat, VerticalFlip)

from utils import reset_tf
from eval_utils import calc_score_variance
from models import  build_seg_model, build_pixel_mlp_class_model
from VegetableSequence import VegetableDataset, VegetableSequence
from temporal_random_seed import TemporalRandomSeed
import myFlags
FLAGS = flags.FLAGS

def main(argv):
    reset_tf(FLAGS.device)
    ds_info = VegetableDataset(FLAGS.data_path)
    dim = ds_info.hsi_dims
    cats = ds_info.get_categories()

    # spoof file path
    assert FLAGS.spoof_type == "print" or FLAGS.spoof_type == "replay"
    spooffn = "224_224.m.rf.npy"
    spoofdir = '03' if FLAGS.spoof_type == 'print' else '04' # "04": replay
    spooffns = [os.path.join(ds_info.DATASET_ROOT_PATH, str(i).zfill(2),
                    "05", spoofdir, spooffn) for i in cats]

    # dataset generation
    input_shape = (224, 224, dim)
    AUGMENTATIONS_ALL = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.2),
        RandomContrast(limit=0.001, p=0.5),
        RandomBrightness(limit=0.001, p=0.5),
        ShiftScaleRotate(
            shift_limit=0.3, scale_limit=0.9,
            rotate_limit=30, border_mode=4, p=0.8),# cv2.BORDER_REFLECT_101
        ToFloat(max_value=1024)
    ])
    AUGMENTATIONS_SIMPLE = Compose([
        ToFloat(max_value=1024)
    ])
    test_aug_gen = VegetableSequence(dataset=ds_info, instance_ids=[5],
                        sample_ids=[1,2], random_state=2, batch_size=32,
                        augmentations=AUGMENTATIONS_ALL, isTest=True)

    # build and load models
    print("building model")
    nb_classes = ds_info.object_categories
    seg_model = build_seg_model(input_shape=input_shape)
    seg_model.load_weights(FLAGS.seg_model)

    pix_class_model = build_pixel_mlp_class_model(
                nb_classes=nb_classes, input_shape=(1,dim))
    pix_class_model.load_weights(FLAGS.class_model)
    penultimate_feat_extractor = Model(inputs=pix_class_model.input,
                outputs=pix_class_model.get_layer("penultimate").output)
    
    def predict_pixel_merge(xs):
        _xs_seg = np.argmax(seg_model.predict(xs), axis=-1)
        assert len(_xs_seg) == len(xs)

        _var_fs = [] # variance of the penultimate features
        for i in range(len(xs)):
            _x = xs[i]
            _x_seg = _xs_seg[i]
            _x_pixels = _x[_x_seg > 0]

            _x_pixels = _x_pixels[:, np.newaxis, :]
            _f_pixels = penultimate_feat_extractor.predict(_x_pixels,
                        batch_size=224*224*dim).reshape(-1, FLAGS.penultimate_nodes)

            _var_f = np.sum(np.var(_f_pixels, axis=0))
            _var_fs.append(_var_f)

        return _var_fs
    predict_func = predict_pixel_merge

    var_fs = []
    true_labels = []
    # process live images
    for i in tqdm.trange(FLAGS.live_augs, desc="live augumentations"):
        for batch in tqdm.tqdm(test_aug_gen, desc="live augumentations batch"):
            xs, ys = batch
            var_f = predict_func(xs)
            var_fs.extend(var_f)
            true_labels.extend(np.argmax(ys, axis=1))

    # process spoof images
    with TemporalRandomSeed(2021):
        for fn in tqdm.tqdm(spooffns, desc="spoofs"):
            x = np.load(fn).astype("uint16")
            xs_aug = np.array([AUGMENTATIONS_ALL(image=x)["image"]
                          for i in range(FLAGS.spoof_augs)])
            var_f = predict_func(xs_aug)
            var_fs.extend(var_f)
            true_labels.extend([10000] * FLAGS.spoof_augs)  # spoof label: 10000

    # calculate accuracy
    true_labels = np.array(true_labels)

    var_fs = np.array(var_fs)
    bin_labels, uncertainties, results = calc_score_variance(true_labels, var_fs)

    # save results
    expr_name = parentdirname(FLAGS.class_model)
    save_result_cache(expr_name, bin_labels, uncertainties, results)
    return 0

def save_result_cache(expr_name, labels, uncertainties, results):
    dn = os.path.join(FLAGS.out_path, expr_name)
    os.makedirs(dn, exist_ok=True)
    np.save(os.path.join(dn, "binary_labels.npy"), labels)
    np.save(os.path.join(dn, "uncertainties.npy"), uncertainties)
    with open(os.path.join(dn, "results.txt"), "w") as f:
        for i, result in enumerate(["TNR95: ", "Detection acc.: ", "ROC: "]):
            f.write(result + str(results[i]) + "\n")
    print("saved to " + dn)

def parentdirname(path):
    return os.path.basename(os.path.dirname(path))

if __name__ == "__main__":
    app.run(main)