# ImgNoise-estimation
A CNN-based system that helps predict additive noise parameters in an image if they are picked up by a detector.

# Theory

Images are affected by a plethora of noise functions that can affect the image in a variety of different ways. Images can be prone to the following types of noise:<br>

<ul>
  <li> Gaussian Noise </li>
  <li> Saltpepper Noise </li>
  <li> Poisson Noise </li>
  <li> Speckle Noise </li>
</ul>
  
## Gaussian Noise

Gaussian Noise presents itself in the following fashion:<br>
<img src="https://www.gergltd.com/cse486/project2/GaussianNoise.jpg" style="width: 500px; max-width: 100%; height: auto" />
<br>
If a kernel must be established for Gaussian Noise then we will get the following distribution:<br>
<img src="https://lh3.googleusercontent.com/proxy/9MtB1mofK0AvWAiJdnkwZYdwzJ3ukcZUyBy-hb6ID1c4nJCfDv_CxVrzByS1WQOStJ8dwPC-czcXUA8PGU9VVzHq-UxUG48u-bBRcrAuMRjfKkbp8MNRtQRn6LzYhUJBXQ3yYC8ChvtMYoPYZQ" style="width: 500px; max-width: 100%; height: auto" />
<br>
Where, 
<ol>
  <li> μ = mean of the distribution ϵ [-1,1] </li>
  <li> σ = variance of the distribution ϵ (0,1] </li>
</ol>

When tinkered with the gaussian noise kernel, the mean in essence controls the brightness of the image and the variance controls the amount of blur apparant in the image. If one wishes to distort an image with gaussian noise, we can use MATLAB's image libraries.

```
img = imnoise(im, 'gaussian') %Gaussian noise with mean=0 and variance=0.01
img = imnoise(im, 'gaussian', mean) %Gaussian noise with mean and variance=0.01
img = imnoise(im, 'gaussian', mean, variance) %Gaussian noise with mean and variance
```

## Saltpepper Noise

Saltpepper noise is exactly how it sounds: a grainy film layered upon your image. Here is what it looks like:<br>
<img src="https://www.researchgate.net/profile/Johan_Van_Niekerk/publication/342314824/figure/fig17/AS:904178192883724@1592584288403/Lena-image-with-dierent-levels-of-salt-pepper-noise-added_Q320.jpg" style="width: 50px; max-width: 100%; height: auto" />
<br>

```
img = imnoise(im, 'saltpepper') %Saltpepper noise affecting 5% of pixels
img = imnoise(im, 'saltpepper', d) %Saltpepper noise affecting (d*100)% of pixels
```

## Speckle Noise

Unlike additive noise, that manifests itself in the following fashion:

```
g(x) = f(x) + n(x)
```
We have speckle noise that manifests itself in a multiplicative fashion:

```
g(x) = f(x) + f(x) * n(x)
```

An example would look like this:<br>
<img src="https://www.researchgate.net/profile/Shuxia_Li3/publication/267810176/figure/fig1/AS:616356856741907@1523962334754/Fig1-original-image-Fig2-speckle-noise-image.png" style="width: 50px; max-width: 100%; height: auto" />
<br>

MATLAB offers speckled output with the same function

```
img = imnoise(im, 'speckle') %Speckle noise affecting the image in the form I+n*I where n~N(0,0.05)
img = imnoise(im, 'speckle', d) %Speckle noise affecting the image in the form I+n*I where n~N(0,d)
```

# Installation

## Get Repository Files

First it is immportant to clone the repository. This can be achieved in the following ways:

### Linux

```
git clone https://github.com/KillingJoke42/ImgNoise-estimation.git
cd ImgNoise-estimation
```

### Windows
For windows one may be able to use above linux method if git bash is installed

```
git clone https://github.com/KillingJoke42/ImgNoise-estimation.git
cd ImgNoise-estimation
```

Under an inavailability of git bash, simply access remote repository via this link: <br>
https://github.com/KillingJoke42/ImgNoise-estimation <br>
Clone zip file in master branch. Unzip the file and change working directory to ImgNoise-estimation <br>

## Get Custom Dataset for Retraining
If one wishes to retrain the models on the dataset used to train the models or explore the dataset used, then one may do so by running the download script provided in the `datasets/` folder. The datasets downloaded are based on three base datasets:
<ol>
  <li> The Berkeley Segmentation Dataset: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/ </li>
  <li> The Pascal VOC 2007 Dataset: http://host.robots.ox.ac.uk/pascal/VOC/ </li>
  <li> USC-SIPI Dataset: http://sipi.usc.edu/database/ </li>
</ol>

To initialize the download, make use of the downnload_dataset module inside the `datasets/` folder
### Linux

```
bash download_dataset.sh
```

### Windows
Windows is not supported as of yet. Next commit will hopefully support automated download for custom dataset. For now, open `download_dataset.sh` in a text editor of your choice and copy the link after curl. Do not include the " > Custom_Dataset_BPU.zip" in the copied string of the link. Once complete, paste the link in a browser to initialize download. Unzip the downloaded .zip file in the `datasets/` folder as is to preserve filetree.

The downloaded dataset should consist of the following filetree:

```
  datasets/
    Berkeley/
      gaussian/
      orig/
      poisson/
      saltpepper/
      speckle/
    Pascal/
      gaussian/
      orig/
      poisson/
      saltpepper/
      speckle/
    USC/
      gaussian/
      orig/
      poisson/
      saltpepper/
      speckle/
```

## Remodeling Original Dataset
If one wishes to remodel the original Berkeley, Pascal and USC images, then the scripts used to create the Custom Dataset is made available in the `scripts/MATLAB` folder. One may also change the dataset path and use different datasets to generate data, should that be required. Code must be changed accordingly, as the MATLAB script is not made with portability and compatibility in mind.

# Usage
## Use Pretrained Models
In order to use the models that were created as part of this project off-the-shelf, each specific application ("detection", "estimation") in `scripts/Python/` has, within itself, a `models/` folder with `*.h5` files inside the directory. These are TensorFlow 2.2.0 model files. One may load the model with the following code and also get the output from it in the following fashion

```
# Code in Python (as of tf2.2.0)
from tensorflow.keras import models
import numpy as np
import cv2
from matplotlib import image

img = image.imread('<IMG_PATH>/<IMAGE_NAME>.<format>')
img = cv2.resize(img, (256, 256))
img = np.resize(img, [1, 256, 256, 1])

model = models.load_model('<MODEL_PATH/MODEL_NAME>.h5')
print(model.predict(img))
```
## Train your own model
Appended with every application folder under `scripts/Python/` is a `*.py` file with all the scripts used to load the dataset, generate the `*.npy` files, train, test and test on a single unseen image the model generated. Once satisfactory training is obtained, one may siphon the model from the `models/` directory in the application root. It is recommended to change the savemodel name in the scripts when retraining to avoid overwrite of the provided default pretrained model.

# Results
After the training, the model has, as of this commit, shown the following performance results:<br>

| Noise Type       | Application          | Accuracy/MAE Loss | Test Acc/Loss | Image Results/Offset |
|------------------|----------------------|-------------------|---------------|----------------------|
| Gaussian Noise   | Detection            | Acc:100%          | Acc:99.5%     | Correct              |
|                  | Parameter Estimation | Loss:0.09         | Loss:0.15     | Offset:0.13          |
| Saltpepper Noise | Detection            | Acc:100%          | Acc:100%      | Correct              |
|                  | Parameter Estimation | Loss:0.01         | Loss:0.03     | Offset:0.02          |
| Speckle Noise    | Detection            | Acc:96.3%         | Acc:95.3%     | Correct              |
|                  | Parameter Estimation | Loss:0.03         | Loss:0.03     | Offset:0.05          |
<br>
