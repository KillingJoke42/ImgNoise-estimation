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

# Scripts

There are two categories of scripts

