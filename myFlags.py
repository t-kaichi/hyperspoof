from absl import flags

# common
flags.DEFINE_string("data_path", "./HSSD/", "path to the dataset")

# model
flags.DEFINE_integer("mlp_nodes", 512, "#nodes in MLPs")
flags.DEFINE_integer("penultimate_nodes", 32, "#nodes in the penultimate layer")
flags.DEFINE_float("loss_weight", 0.0002, "weight of variance loss")

# train
flags.DEFINE_integer("device", 0, "visible device")
flags.DEFINE_string("log_root", "./logs", "log directory")
flags.DEFINE_bool("isSeg", False, "segmentation or classification")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_integer("nb_pixels", 1024, "#pixels from a sample")
flags.DEFINE_integer("epochs", 5000, "max epochs")
flags.DEFINE_integer("patience", 1000, "patience of the early stopping")

# test
flags.DEFINE_integer("live_augs", 4, "#augmentations of lives")
flags.DEFINE_integer("spoof_augs", 8, "#augmentations of spoofs")
flags.DEFINE_string("spoof_type", "print", "type of spoof attack, print or replay")
flags.DEFINE_string("seg_model", "logs/HSSD-seg-1624333422/best.weights.hdf5",
            "path of hsi segmentation model weights")
flags.DEFINE_string("class_model", "logs/HSSD-pix_class-1624333932/best.weights.hdf5",
            "path of pixel classification model weights")
flags.DEFINE_string("out_path", "./out/", "output dir")
