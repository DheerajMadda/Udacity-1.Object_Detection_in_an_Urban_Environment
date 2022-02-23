# Object detection in an Urban Environment

This is a repository containing the details of the Computer Vision starter module for Udacity Self Driving Car ND.


## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/). The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. 

## Structure

The data in the classroom workspace will be organized as follows:
```
/home/backups/
    - raw: contained the tf records in the Waymo Open format. (NOTE: this folder only contains temporary files and should be empty after running the download and process script)

/home/workspace/data/
    - processed: contained the tf records in the Tf Object detection api format. (NOTE: this folder should be empty after creating the splits)
    - test: contain the test data
    - train: contain the train data
    - val: contain the val data
```

The experiments folder will be organized as follow:
```
experiments/
    - exporter_main_v2.py: to create an inference model
    - model_main_tf2.py: to launch training
    - experiment0/....
    - experiment1/....
    - experiment2/...
    - pretrained-models/: contains the checkpoints of the pretrained models.
```

## Instructions

### Download and process the data

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you will need to implement the `create_tf_example` function. This function takes the components of a Waymo Tf record and save them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file. 

Once you have coded the function, you can run the script at using
```
python download_process.py --data_dir /home/workspace/data/ --temp_dir /home/backups/
```

You are downloading XX files so be patient! Once the script is done, you can look inside the `/home/workspace/data/processed` folder to see if the files have been downloaded and processed correctly.


### Exploratory Data Analysis

Now that you have downloaded and processed the data, you should explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation). 


### Create the splits

Now you have become one with the data! Congratulations! How will you use this knowledge to create the different splits: training, validation and testing. There are no single answer to this question but you will need to justify your choice in your submission. You will need to implement the `split_data` function in the `create_splits.py` file. Once you have implemented this function, run it using:
```
python create_splits.py --data_dir /home/workspace/data/
```

NOTE: Keep in mind that your storage is limited. The files should be <ins>moved</ins> and not copied. 

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf). 

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `training/pretrained-models/`. 

Now we need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 4 --checkpoint ./training/pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Create a folder `training/reference`. Move the `pipeline_new.config` to this folder. You will now have to launch two processes: 
* a training process:
```
python model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config
```
* an evaluation process:
```
python model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config --checkpoint_dir=training/reference/
```

NOTE: both processes will display some Tensorflow warnings.

To monitor the training, you can launch a tensorboard instance by running `tensorboard --logdir=training`. You will report your findings in the writeup. 

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup. 

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it. 


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:
```
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path training/experiment0/pipeline.config --trained_checkpoint_dir training/experiment0/ckpt-50 --output_directory training/experiment0/exported_model/
```

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py -labelmap_path label_map.pbtxt --model_path training/experiment0/exported_model/saved_model --tf_record_path /home/workspace/data/test/tf.record --config_path training/experiment0/pipeline_new.config --output_path animation.mp4
```

## Submission Template

### Project overview

This is a repository collecting information about the Udacity Self-Driving Car project, in which we used the TensorFlow object detection API to improve the detection of things such as cars, pedestrians, and bicycles. This repository offers instructions for downloading the tfrecord sample files from Cloud storage and splitting them for object detection API training. 

The dataset used for this purpose is [Waymo](https://waymo.com/open/) which can be downloaded from the [Google Cloud Storage Bucket]((https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/). In this case, we will be using tfrecord files which we will be modified into tf.Train.Example for the object detection api format. We will also be splitting the dataset into training, validation and testing sets using np.split in  "create_splits.py" python program.  

### Set up

- The following libraries can be installed

```
pip install tensorflow-gpu==2.3.0
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
```



### Dataset

We must fit rectangular bounding boxes on photos with objects, such as walkers, cyclists, and autos, in the dataset. Images were taken in various locations, under various weather conditions, and at various times of the day (day/night). The image set includes a variety of photographs, some of which are hazy, clear, light, and dark. Below is a sample image with a dark and foggy background.

<img src="https://i.imgur.com/c3F958n.png">

<img src="https://i.imgur.com/TJ8Ea3C.png">



#### Dataset analysis

The majority of the labels in the dataset are connected with autos and pedestrians, with a minor sample size of bikes. The "Exploratory Data Analysis.ipynb" file contains the proportionate quantity of counts for each label. There is a disparity in class that can be addressed using oversampling techniques. Below is a sample image of the proportional counts of the labels (cars, pedestrians, and cyclists):

<img src="https://i.imgur.com/8jtDAmz.png">

Images were captured in a variety of settings (subway, highway, city), under various weather conditions (foggy/sunny), and at various times of the day (day/night). Vehicles have red bounding boxes, cyclists have green bounding boxes, while pedestrians have blue bounding boxes.


Further examination of the dataset reveals that the bulk of the photographs involve vehicles and pedestrians (majority vehicles), with only a few sample images containing bikes. A bar plot of the distribution of classes (automobiles, pedestrians, and cyclists) over a collection of 20000 random photos in the dataset is shown in the chart below. In the "Exploratory Data Analysis.ipynb" notebook, the analysis is updated.

We noticed that there are extremely few cyclists in the photographs. Only about 2400 photos feature at least one cyclist, while the maximum number of cyclists present in each image is only 6.


#### Cross validation

We're working with a total of 100 tfrecord files. The data is initially shuffled at random, then divided into training, testing, and validation sets. Random shuffling is used to balance out the class imbalance in each sample. The shuffling guarantees that samples in the training, testing, and validation datasets are distributed evenly.

Because we are only using 100 tfrecord samples, we are choosing 0.75: 0.15 as the percentage of training and validation data. This guarantees that we have enough data for both training and validation. To examine the error rate and see if the model is overfitting, we use 10% (0.1) of the sample as the test set. Overfitting should not be an issue because 75% of the time is spent training and the remaining 10% is spent testing.

### Training 

#### Reference experiment

The residual network model ([Resnet](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)) without augmentation , model loss is shown below:

<img src="https://i.imgur.com/vC9whPX.jpg">

The model was initially overfitting because the training and validation losses diverged. The validation loss is shown in blue and the training loss is shown in orange. This divergence suggests a high rate of model validation errors, indicating that the model is overfitting.
The precision and recall curves show that the model's performance gradually improves as precision and recall both increase. A high recall rate isn't always desirable, and the model's performance isn't always excellent.
Precision:

<img src="https://i.imgur.com/z4hSFrv.jpg">

Recall:

<img src="https://i.imgur.com/e3rRSdH.jpg">


#### Improve on the reference

The initial step in improving the model's performance was to enhance the photos by converting them to grayscale with a probability of 0.2. Following that, we clamped the contrast values between 0.6 and 1.0 to allow for additional lighting datapoints to be classified. Because a larger portion of the photos were darker, boosting the brightness to 0.3 created an even datapoint that the model could better classify. "'pipeline new.config"' contains the pipeline updates.

Augmentations applied:

- 0.02 probability of grayscale conversion
- brightness adjusted to 0.3
- contrast values between 0.6 and 1.0


The details of the run can be found here : "Explore augmentations.ipynb"

The model loss with augmentation :

<img src="https://i.imgur.com/H4DtUd8.jpg">

Precision with Augmentation:

<img src="https://i.imgur.com/2aGRa93.jpg">

Recall with Augmentation:

<img src="https://i.imgur.com/wtTj62o.jpg">
The loss is lesser than the one before it (un-augmented model). This is a sign of improved performance. More samples of augmented datapoints, such as merging contrast values with grayscale, should be included. Instead of setting the brightness to 0.3, it can be set to a specific value.
The most significant aspect is to increase the number of bikes and pedestrians in the dataset, which are now in short supply. This is an unavoidable need since model biases play a significant part in loss curves, and the less variety in training samples, the worse the accuracy.

With augmentation, we were able to limit overfitting to some extent, but a more balanced dataset would yield better classification results.
