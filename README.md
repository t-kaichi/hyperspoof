# Hyperspoof
The official implementation of [`A Hyperspectral Approach for Unsupervised Spoof Detection with Intra-sample Distribution`](https://ieeexplore.ieee.org/document/9506625) (ICIP2021).  
The hyperspectral spoofing dataset (HSSD) is available [here](https://drive.google.com/drive/folders/1OBQsfNAhdBHqk0o1MRdqeU6fDeqEOV-7?usp=sharing).
Please read carefully the Data License before you download the HSSD.

## Dependencies
- Python3
- TensorFlow >= 2.4  

This code has been developed with Docker container created from `nvcr.io/nvidia/tensorflow:21.03-tf2-py3`.  
See `requirements.txt` for additional dependencies and version requirements.
```
pip install -r requirements.txt
```

## Training
Download HSSD from [link](https://drive.google.com/drive/folders/1OBQsfNAhdBHqk0o1MRdqeU6fDeqEOV-7?usp=sharing).
Afterwards, set the --data_path argument to the corresponding extracted HSSD path.

Train single-pixel classifier.
```
python train.components.py --data_path ./HSSD
```

Train foreground extractor.
```
python train.components.py --isSeg --data_path ./HSSD
```

## Evaluation
Set the --class_model and --seg_model arguments to the corresponding learned single-pixel classifier and foreground extractor, respectively.
```
python test.py --data_path ./HSSD --class_model ./logs/HSSD-pix_class-0000/best.weights.hdf5 --seg_model ./logs/HSSD-seg-0000/best.weights.hdf5
```

## Citation
```
@INPROCEEDINGS{9506625,
  author={Kaichi, Tomoya and Ozasa, Yuko},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)}, 
  title={A Hyperspectral Approach For Unsupervised Spoof Detection With Intra-Sample Distribution}, 
  year={2021},
  pages={839-843},
  doi={10.1109/ICIP42928.2021.9506625}
}

```
