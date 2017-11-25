Inception U-Net: Ultrasound Nerve Segmentation
=============================

Semantic image segmentation of a collection of nerves called Brachial Plexus in ultrasound images


### Environment Requirements

The following needs to be installed:


- Python >= 3.6
- numpy >= 1.13
- Tensorflow >= 1.3
- Keras >= 2.0
- OpenCV >= 3.1
- tqdm >= 4.19


NOTE: tqdm can be installed using ```pip install tqdm```


### Downloading the Dataset

The dataset can be downloaded from Kaggle competetion "Ultrasound Nerve Segmation":

https://www.kaggle.com/c/ultrasound-nerve-segmentation/data

Download the ***train*** set and unzip the contents. Put the images (.tif files) in ```<root_project_dir>/data/train/```.


### Preprocessing
After downloading the dataset, run the preprocess routine by issuing the command:

```bash
python preprocess.py
```

After running the script, the following files will be generated:
```
data/train_xs96/        - Directory containing the filtered and resized images
data/train_set.npz      - Train set in numpy readable format
data/validation_set.npz - Validation set in numpy readable format
data/train_stats.pkl    - Pickle file that contains basic statistics about the train set
```



### Training the Model

To train the model, execute the command:

```bash
python train.py
```

Training takes approximately 63 seconds per epoch on Amazon AWS p2.xlarge instance with GPU support. A pre-trained model, that was ran for 120 epochs, can be downloaded at:

https://www.dropbox.com/s/as0cj9uv8imf4nb/weights.h5?dl=0

Place the ***'weights.h5'*** ```output/``` directory.



### Running Inference

To run inference, place the target images (.tif files) in a directory. If the images have accompanying ground truth labels, the labels must have the same name as the target and end with ```<name>_mask.tif```.

Execute the command:

```bash
python inference.py -s <image_dir>
```
The generated output will be saved in ```<image_dir>_output```

For more details on accepted arguments, see ```inference.py``` file.


A sample set of images can be found at ```sample_data/``` directory. To see a smaple output, execute the command:

```bash
python inference.py -s sample_data/
```

The generated output can be found at ```sample_data_output/```.