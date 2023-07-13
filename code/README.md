## Image Captioning

**Contributors (alphabetically):**  Quan Qi, Yuan Shang, Kun Yu, Runfeng Zhang

Image Captioning is the process of generating textual descriptions of an
image. The most common deep-learning-based solutions address this task with a visual-encoder/NLP-decoder architecture. In this project, we explored different visual-encoder and NLP-decoder structures, and test their performances on COCO2014 datasets. To make our models comparable, in the baseline architecture (Model1), We used RESNET101 for the visual-encoder, and LSTM for the NLP-decoder with attention mechanisms. This baseline model have a bleu4=0.3372 in the evaluation dataset in our best-trained model.

For the visual-encoder module, in recent years, several non-CNN-based structures such as MLPMixer and ViT, which also show good performance in computer vision tasks. We applied MLPMixer as an alternative visual-encoder (Model2), and it showed competing performance with the baseline architecture based on its bleu4=0.2996.

For the NLP\-decoder module, we first tested the performance without attention mechanisms(Model3, bleu4=0.3388). Then, we applied transformers. A transformer includes a sub-encoder and a sub-decoder part. We implemented different transformer structures, including 1. transformer with only the sub-encoder part (Model4,bleu4=0.1399); 2. transformer with only the sub\-decoder part (Model5,bleu4=0.3210); 3. transformer with both sub-encoder and sub-decoder parts (Model6,bleu4=0.3307).

Finally, we compared the pros and cons of all our models and discussed how these factors might affect our final performance.

**Acknowledgements:** 

* Thanks [sgrvinod](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)'s code as a foundation of our code.
* Thanks [RoyalSkye](https://github.com/RoyalSkye/Image-Caption/tree/e528b36b32fdc8175921ce60bb9a2c6cecafebb8) for the transformer code. and README template.

### Section 1: Run

#### 1.1 Dataset

Images: We are using the **MSCOCO'14** Dataset: [Training (13GB)](http://images.cocodataset.org/zips/train2014.zip) and [Validation (6GB)](http://images.cocodataset.org/zips/val2014.zip)

Caption: We will use [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip).

#### 1.2 Environment Set Up

We are using Python 3.10,torch==1.13.1 and torchvision==0.14.1.

Create environment using **environment.yaml**
Conda environment tutorial(https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
```shell
conda env create -f environment.yaml
```

#### 1.3 Modules

**create_input_files.py**

Before training, you should run this module to generate required data files. (10 files)

> * TEST/TRAIN/VAL_CAPLENS_X.json: the length of each caption.
>* TEST/TRAIN/VAL_CAPTIONS_X.json: all captions of images.
> * TEST/TRAIN/VAL_IMAGES_X.hdf5: image data, stored in hdf5 format.
>* WORDMAP_X.json: A dictionarty that converts words into id.

Parameters:

>* --dataset: which dataset we used, either 'coco', 'flickr8k' or 'flickr30k'.
>* --karpathy_json_path: path of captions dataset.
>* --image_folder: path of image dataset.
>* --captions_per_image: how many captions each image has?
>* --min_word_freq: the frequency of words less than it will be <unk>.
>* --output_folder: output file path.
>* --max_len: the maximum length of each caption.

**train.py**

In this module, you can easily check or modify the parameters you wish to. You can train your model from scratch, or resume training at a checkpoint by point to the corresponding file with the checkpoint parameter. It will also perform validation at the end of every training epoch.

 This module has too many parameters, so we only introduce parts of them.

You can modify model parameters to change the structure of the network. In our implementation, there are two attention methods: `ByPixel` and `ByChannel`, and two decoder modes: `lstm` and `transformer`. Training parameters are used to control the hyperparameters of learning such as learning rate and batch size. "alpha_c" is the weight assigned to the second loss, and checkpoint is the path of the previous model. All parameters are shown in below table.

|      **Type**       | **Parameters**                                                                                                                                                                                                   |
| :-----------------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset parameters  | -data_folder; --data_name;                                                                                                                                                                                       |
|  Model parameters   | --emb_dim; --attention_dim; --decoder_dim; --n_heads; </br>--dropout;  --decoder_mode; --attention_method; </br>--encoder_layers; --decoder_layers;                                                              |
| Training parameters | --epochs; --stop_criteria; --batch_size; --print_freq; </br>--workers; --encoder_lr; --decoder_lr; --grad_clip; </br>--alpha_c; --checkpoint; --fine_tune_encoder;</br> --fine_tune_embedding; --embedding_path; |

**datasets.py**

A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.

**models_X.py**

The PyTorch neural network class of different architectures. 'X" equals to 1-6, refering to our Model1-6.

**training_X.py**
Training codes for models_X.py. Since the loss functions are a little different, and the hyper-parameters might also need to be tuned for each model. We have a separate training.py for each Model to gave us the maximum flexibility so that we could tune and run our models separately. During validation phase of each epoch, the output of the 'guided/teacher-forcing' top caption is compared with the true caption to get the loss. 

**eval_X.py**

Evaluating the trained model by generating the caption based on Beam Search given a searchsize=5.  

Parameters:

>* --data_folder: folder with data files saved by create_input_files.py.
>* --data_name: base name shared by data files.
>* --decoder_mode: which network does the decoder of the model use?
>* --beam_size: beam search hyperparameters.
>* --checkpoint: the path of a trained model.

**caption_X.py**

Read an image and caption it with beam search. The output of this module is a generated caption and the visualization of attention.

Parameters:

> * --img: path of images, it can be a single img file or a folder filled with img.
> * --checkpoint: the path of a trained model.
> * --word_map: path of wordmap json file generated by create_input_files.py.
> * --decoder_mode: which network does the decoder of the model use?
> * --save_img_dir: path to save the output file.
> * --beam_size: beam search hyperparameters.
> * --dont_smooth: do not smooth alpha overlay.

**utils.py**

This module consists of some useful functions, such as save checkpoints, write log files, update learning rates, log training matrix. Those functions are called many times. To keep the code clean, we move them into this module.

#### 1.4 Run

To run, make sure we have the right directory in create_input_files.py, train.py, eval.py, and caption.py.

In create_input_files.py:
```python
    create_input_files(dataset='coco',
                       karpathy_json_path='data/caption_datasets/dataset_coco.json',
                       image_folder='D:/data',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='D:/data/output',
                       max_len=50)
```

In train_X.py,line 15-51:
```python
data_folder = 'data/output'  # the output folder of 'create_input_files.py'
checkpoint_folder = 'data/checkpoint'
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
```

If we want to resume training from a checkpoint:
```python
#'BEST_checkpoint_model2_coco_5_cap_per_img_5_min_word_freq.pth.tar'
# We could also resume from your last checkpoint, which is also saved during training
checkpoint =  'BEST_checkpoint_model2_coco_5_cap_per_img_5_min_word_freq.pth.tar'
```

In caption_X.py line 354:
```python
    args = Bar({"img": 'myimg/dog.jpg',
                "model": 'data/checkpoints/checkpoint_model1_coco_5_cap_per_img_5_min_word_freq.pth.tar',
                "word_map": 'data/output/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json',
                "beam_size": 5,
                "smooth": False
                })
```

In eval_X.py:
```python
data_folder = 'data/output' # 'caption data'
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
# model checkpoint
checkpoint = 'data/checkpoint/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # checkpoints/
# word map, ensure it's the same the data was encoded with and the model was trained with
word_map_file = 'data/output/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
```

After we have change the directories, activate our environment and cd to the model's directory.

First, we will create the input files.
```shell
$ python create_input_files.py
```

Then we will train the model. 
```shell
$ python train_X.py
```

To evalute the model: 
```shell
$ python eval_X.py
```

To create caption of an image. 
```shell
$ python eval_X.py
```

This will create a captioned image in the directory we specified. Make sure we put an image in the myimg folder. For example, I put a image named 'dog.jpg' in the folder to create caption for it.
```python
    args = Bar({"img": 'myimg/dog.jpg',
                "model": 'data/checkpoints/checkpoint_model1_coco_5_cap_per_img_5_min_word_freq.pth.tar',
                "word_map": 'data/output/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json',
                "beam_size": 5,
                "smooth": False
                })
```

Please note that there are a lot of redundencies in our codes. We do that this way so tuning of Model1-6 will more flexible. A more care refactoring of our code will make them looks more beautiful.
