def set_seed():
    import numpy as np
    import tensorflow as tf
    import random
    # Fix random seeds
    SEED = 2021
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)


def reset_tf(gpu_id=0):
    # Setup tf
    import tensorflow as tf
    if tf.__version__ >= "2.1.0":
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
        print("set memory growth GPU No.: {}".format(gpu_id))
    elif tf.__version__ >= "2.0.0":
        #TF2.0
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(physical_devices[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)
        physical_devices = tf.config.list_physical_devices('GPU')
        print("set memory growth GPU No.: {}".format(gpu_id))
    else:
        raise NotImplementedError("fill out process for TensorFlow ver. {}".format(tf.__version__))
        
    return 0 

def hyper2truecolor(hyper, min_l=400.0, res=5, scale=1, isChannelsFirst=False):
    import numpy as np
    # (default): [665.73, 557.07, 478.17] nm
    # https:#github.com/davidkun/HyperSpectralToolbox/search?q=rgb&unscoped_q=rgb
    bgr_l = [665.73,  # for r
             557.07,  # for g
             478.17]  # for b
    bgr_l.reverse()

    bands = [int((l - min_l) / res) for l in bgr_l]

    if isChannelsFirst:
        bgr_img = np.empty(hyper.shape[1:] + (3, ))
        for i, band in enumerate(bands):
            bgr_img[:, :, i] = hyper[band, :, :]
    else:
        bgr_img = np.empty(hyper.shape[:2] + (3, ))
        for i, band in enumerate(bands):
            bgr_img[:, :, i] = hyper[:, :, band]
    return bgr_img * scale

def hyper2rgb_batch(xs):
    import numpy as np
    rgbs = []
    for hyperImg in xs:
        denoised = np.where(hyperImg > 1024, 0, hyperImg)
        max_pixval = np.amax(denoised)
        hyperImg *= 255.0 / max_pixval
        rgbImg = hyper2truecolor(hyperImg, scale=1).astype("uint8")
        rgbs.append(rgbImg)
    return np.array(rgbs)