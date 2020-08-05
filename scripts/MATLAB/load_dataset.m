% Create dataset for nose classifier
% Author: Anant Raina
% Date: 2nd August, 2020

% There are 8 types of noise:
% 1) White noise. Additive white Gaussian noise.
% 2) Black noise.
% 3) Gaussian noise.
% 4) Pink noise or flicker noise, with 1/f power spectrum.
% 5) Brownian noise, with 1/f2 power spectrum.
% 6) Contaminated Gaussian noise, whose PDF is a linear mixture of Gaussian PDFs.
% 7) Power-law noise.
% 8) Cauchy noise.

%This code is used to generate a dataset that can help feed into the deep learning algorithm
%that will detect the type of noise.
% J = imnoise(I,'gaussian',m,var_gauss)
% J = imnoise(I,'poisson') (Last To Attempt)
% J = imnoise(I,'salt & pepper',d)
% J = imnoise(I,'speckle',var_speckle)

%We will be using three datasets
%1) USC SIPI dataset
%2) Pascal VOC 2007 dataset
%3) Berkeley Segmentation dataset

%% Dataset FLAGS
%Define PATHS
% b = berkeley; p = pascal; u = usc

%dataset PATH
b_parent = "/home/killingjoke42/projects/ip_special/datasets/Berkeley/orig";
p_parent = "/home/killingjoke42/projects/ip_special/datasets/Pascal/orig";
u_parent = "/home/killingjoke42/projects/ip_special/datasets/USC/orig";

%out PATH
b_out = "/home/killingjoke42/projects/ip_special/datasets/Berkeley/";
p_out = "/home/killingjoke42/projects/ip_special/datasets/Pascal/";
u_out = "/home/killingjoke42/projects/ip_special/datasets/USC/";

%dataset fileformat
b_fileformat = ".jpg.bmp";
p_fileformat = ".bmp";
u_fileformat = ".bmp";

b_imagecount = 200;
p_imagecount = 200;
u_imagecount = 100;

%% Perform Gaussian Conversion

%Berkeley Modifier for Gaussian
for i=1:10
    for j=1:b_imagecount
        img = imresize(imread(b_parent + "/" + j + b_fileformat), [256 256]);
        rand_mean = -1 + (1+1)*rand();
        rand_var = rand();
        imwrite(imnoise(img, "gaussian", rand_mean, rand_var), b_out+"gaussian/"+(200*(i-1)+j)+"_m"+rand_mean+"_v"+rand_var+"_.bmp");
    end
end

%Pascal Modifier for Gaussian
for i=1:10
    for j=1:p_imagecount
        img = imresize(imread(p_parent + "/" + j + p_fileformat), [256 256]);
        rand_mean = -1 + (1+1)*rand();
        rand_var = rand();
        imwrite(imnoise(img, "gaussian", rand_mean, rand_var), p_out+"gaussian/"+(200*(i-1)+j)+"_m"+rand_mean+"_v"+rand_var+"_.bmp");
    end
end

%USC Modifier for Gaussian
for i=1:20
    for j=1:u_imagecount
        img = imresize(imread(u_parent + "/" + j + u_fileformat), [256 256]);
        rand_mean = -1 + (1+1)*rand();
        rand_var = rand();
        imwrite(imnoise(img, "gaussian", rand_mean, rand_var), u_out+"gaussian/"+(100*(i-1)+j)+"_m"+rand_mean+"_v"+rand_var+"_.bmp");
    end
end

%% Perform Salt and Pepper Conversion

%Berkeley Modifier for SAP
for i=1:10
    for j=1:b_imagecount
        img = imresize(imread(b_parent + "/" + j + b_fileformat), [256 256]);
        rand_d = rand();
        imwrite(imnoise(img, "salt & pepper", rand_d), b_out+"saltpepper/"+(200*(i-1)+j)+"_d"+rand_d+"_.bmp");
    end
end

%Pascal Modifier for SAP
for i=1:10
    for j=1:p_imagecount
        img = imresize(imread(p_parent + "/" + j + p_fileformat), [256 256]);
        rand_d = rand();
        imwrite(imnoise(img, "salt & pepper", rand_d), p_out+"saltpepper/"+(200*(i-1)+j)+"_d"+rand_d+"_.bmp");
    end
end

%USC Modifier for SAP
for i=1:20
    for j=1:u_imagecount
        img = imresize(imread(u_parent + "/" + j + u_fileformat), [256 256]);
        rand_d = rand();
        imwrite(imnoise(img, "salt & pepper", rand_d), u_out+"saltpepper/"+(100*(i-1)+j)+"_d"+rand_d+"_.bmp");
    end
end

%% Perform spekle conversion

%Berkeley Modifier for Speckle
for i=1:10
    for j=1:b_imagecount
        img = imresize(imread(b_parent + "/" + j + b_fileformat), [256 256]);
        rand_var = rand();
        imwrite(imnoise(img, "speckle", rand_var), b_out+"speckle/"+(200*(i-1)+j)+"_v"+rand_var+"_.bmp");
    end
end

%Pascal Modifier for Speckle
for i=1:10
    for j=1:p_imagecount
        img = imresize(imread(p_parent + "/" + j + p_fileformat), [256 256]);
        rand_var = rand();
        imwrite(imnoise(img, "speckle", rand_var), p_out+"speckle/"+(200*(i-1)+j)+"_v"+rand_var+"_.bmp");
    end
end

%USC Modifier for Speckle
for i=1:20
    for j=1:u_imagecount
        img = imresize(imread(u_parent + "/" + j + u_fileformat), [256 256]);
        rand_var = rand();
        imwrite(imnoise(img, "speckle", rand_var), u_out+"speckle/"+(100*(i-1)+j)+"_v"+rand_var+"_.bmp");
    end
end

%% Perform salt and pepper conversion

%Berkeley Modifier for SAP
for i=1:10
    for j=1:b_imagecount
        img = imresize(imread(b_parent + "/" + j + b_fileformat), [256 256]);
        rand_d = rand();
        imwrite(imnoise(img, "salt & pepper", rand_d), b_out+"saltpepper/"+(200*(i-1)+j)+"_d"+rand_d+"_.bmp");
    end
end

%Pascal Modifier for SAP
for i=1:10
    for j=1:p_imagecount
        img = imresize(imread(p_parent + "/" + j + p_fileformat), [256 256]);
        rand_d = rand();
        imwrite(imnoise(img, "salt & pepper", rand_d), p_out+"saltpepper/"+(200*(i-1)+j)+"_d"+rand_d+"_.bmp");
    end
end

%USC Modifier for SAP
for i=1:20
    for j=1:u_imagecount
        img = imresize(imread(u_parent + "/" + j + u_fileformat), [256 256]);
        rand_d = rand();
        imwrite(imnoise(img, "salt & pepper", rand_d), u_out+"saltpepper/"+(100*(i-1)+j)+"_d"+rand_d+"_.bmp");
    end
end

%% Perform poisson conversion

%Berkeley Modifier for poisson
for j=1:b_imagecount
    img = imread(b_parent + "/" + j + b_fileformat);
    imwrite(imnoise(img, "poisson"), b_out+"poisson/"+(200*(i-1)+j)+"_.bmp");
end

%Pascal Modifier for poisson
for j=1:p_imagecount
    img = imread(p_parent + "/" + j + p_fileformat);
    imwrite(imnoise(img, "poisson"), p_out+"poisson/"+(200*(i-1)+j)+"_.bmp");
end

%USC Modifier for poisson
for j=1:u_imagecount
    img = imread(u_parent + "/" + j + u_fileformat);
    imwrite(imnoise(img, "poisson"), u_out+"poisson/"+(100*(i-1)+j)+"_.bmp");
end