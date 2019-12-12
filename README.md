## Super Resolution Examples


This SRGAN is forked from https://github.com/tensorlayer/srgan

### SRGAN Architecture

TensorFlow Implementation of ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802) 

from the paper:

>    ...To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by perceptual similarity instead of similarity in pixel space...

This investigation adds to the descriminator the difference between the LR version of the generated image from the input LR image as well. 

<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="img/model.jpeg" width="80%" height="10%"/>
</div>
</a>


### Results

<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<!--<img src="img/SRGAN_Result2.png" width="80%" height="50%"/> -->
</div>
</a>

<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<!--<img src="img/SRGAN_Result3.png" width="80%" height="50%"/> -->
</div>
</a>

### Prepare Data and Pre-trained VGG

- 1. You need to download the pretrained VGG19 model in [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs).
- 2. You need to have the high resolution images for training.
  -  In this experiment, I used images from [DIV2K - bicubic downscaling x4 competition](https://data.vision.ee.ethz.ch/cvl/DIV2K/), so the hyper-paremeters in `config.py` (like number of epochs) are seleted basic on that dataset, if you change a larger dataset you can reduce the number of epochs. 
  -  If you want to use your own images, you can set the path to your image folder via `config.TRAIN.hr_img_path` in `config.py`.



### Run
- Set your image folder in `config.py`, if you download [DIV2K - bicubic downscaling x4 competition](http://www.vision.ee.ethz.ch/ntire17/) dataset, you don't need to change it. 
- Other links for DIV2K, in case you can't find it : [test\_LR\_bicubic_X4](https://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_test_LR_bicubic_X4.zip), [train_HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip), [train\_LR\_bicubic_X4](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip), [valid_HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/validation_release/DIV2K_valid_HR.zip), [valid\_LR\_bicubic_X4](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip).

```python
config.TRAIN.img_path = "your_image_folder/"
```

- Start training.

```bash
python train.py
```

- Start evaluation. 

<!--([pretrained model](https://github.com/tensorlayer/srgan/releases/tag/1.2.0) for DIV2K)-->

```bash
python train.py --mode=evaluate 
```

### citations 
@InProceedings{Agustsson_2017_CVPR_Workshops,
	author = {Agustsson, Eirikur and Timofte, Radu},
	title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
	month = {July},
	year = {2017}
} 

@InProceedings{Timofte_2017_CVPR_Workshops,
author = {Timofte, Radu and Agustsson, Eirikur and Van Gool, Luc and Yang, Ming-Hsuan and Zhang, Lei and Lim, Bee and others},
title = {NTIRE 2017 Challenge on Single Image Super-Resolution: Methods and Results},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {July},
year = {2017}
}

@InProceedings{Timofte_2018_CVPR_Workshops,
author = {Timofte, Radu and Gu, Shuhang and Wu, Jiqing and Van Gool, Luc and Zhang, Lei and
Yang, Ming-Hsuan and Haris, Muhammad and others},
title = {NTIRE 2018 Challenge on Single Image Super-Resolution: Methods and Results},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2018}
}

@InProceedings{Timofte_2018_CVPR_Workshops,
author = {Timofte, Radu and Gu, Shuhang and Wu, Jiqing and Van Gool, Luc and Zhang, Lei and
Yang, Ming-Hsuan and Haris, Muhammad and others},
title = {NTIRE 2018 Challenge on Single Image Super-Resolution: Methods and Results},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2018}
}

@InProceedings{Ignatov_2018_ECCV_Workshops,
author = {Ignatov, Andrey and Timofte, Radu and others},
title = {PIRM challenge on perceptual image enhancement on smartphones: report},
booktitle = {European Conference on Computer Vision (ECCV) Workshops},
month = {January},
year = {2019}
}


### Reference
* [1] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

### Author
- [Tenfleques](https://github.com/tenfleques)

### License

- For academic and non-commercial use only.
