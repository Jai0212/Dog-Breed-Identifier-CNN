# Dog Breed Identification Using CNN

This is a deep learning neural network (CNN) built from scratch that identifies the breed of a dog from its image.  
The deep learning model is designed to have 120 dog breeds in its data.

The dataset involved a total of 20,580 images including 12,000 training images and 8,580 testing images.  

The model has a training accuracy of ~98% and ~3% loss. The model is designed using a 5-layered neural network.  
It uses 2D convolutions and 2D max pooling with varying filters as its primary layers with dropouts in between.  
The CNN ends with a flattening layer and dense layers accompanied by batch normalization.  
The entire module is then compiled using the 'adam' optimizer and runs for 10 epochs.  

This model can then be used to predict the breed of a dog from its image.

The dataset of dogs was obtained from http://vision.stanford.edu/aditya86/ImageNetDogs/. 

<img width="706" alt="Screenshot 2024-01-25 at 2 11 58 AM" src="https://github.com/Jai0212/Dog-Breed-Identifier-CNN/assets/86296165/109328e5-b613-4884-a5a3-e848f0878d16">
The above are the predictions made by the CNN for different breeds of dogs.

## Features
* The CNN model uses 5 primary layers for image detection. 
* This model uses over 20,000 images with layers specially designed to obtain the best results.  
* The model is trained to differentiate between 120 dog breeds.  
* The program is customizable for different datasets and models as well.
* Despite the large number of labels, the model has a training accuracy of ~98% and ~3% loss. 
* The testing accuracy is ~80% with a small number of dog breeds but this can be increased for the larger class 
size with more layers and an overall refined model (which requires more time and resources)
* It is easy to use and understand for beginners with a proper explanation for code in the python files. 
* The model is also not too heavy and is thus able to run on a normal laptop efficiently, avoiding any out of memory 
or runtime errors and still providing quite accurate results.

<img width="1222" alt="Screenshot 2024-01-25 at 2 22 07 AM" src="https://github.com/Jai0212/Dog-Breed-Identifier-CNN/assets/86296165/e7f99195-de82-454c-8b62-22f5ffa79640">

<img width="952" alt="Screenshot 2024-01-18 at 2 01 32 AM" src="https://github.com/Jai0212/Dog-Breed-Identifier-CNN/assets/86296165/7e96801b-0918-4f1f-ae81-8337e08d57a3">

The graphs above show the accuracy and loss graphs with 10 epochs (y axis). The last image is the console log during the
training of the model.  


## Technical Aspects
The entire code was written in Python (3.11) on Pycharm. The packages/dependencies used were:
- tensorflow
- numpy 
- opencv-python 
- os
- scipy
- matplotlib


Files Explanation:  
* The **main.py** file is for gpu allocation and generating the dog breed names.  
* The **model_creator.py** file is where the entire code is there for reading the images and the .mat files and using the
data to create the CNN model.  
* The **model_accurate_loss_data.py** file is where the code for generating the loss and accuracy data is. In case you 
modify the model you can check its statistics from this file.  
* The **user.py** file is for the user interface which allows the user to test the model by uploading images.  
* The user will upload the images under **'images_to_be_identified_user'**.  
* The images of the dogs (dataset) are under **'data'**. Under 'data' there are folders for each dog breed containing 
images for each dog breed.  
* The model that I have created is stored under **'models'** as a .h5 file.  
* The .mat files, which contain the data for whether the image is for testing or training are under **mat_files**.  


## Installation
The program is designed such that anyone can use it. One needs to download the files and open them in any Python IDE
(it was created using PyCharm). Once the files are in the IDE, you must download the following packages/dependencies
in order for the code to run - numpy, opencv-python, tensorflow, os, scipy and matplotlib. Although not all of these
packages are necessary to download, I would recommend them in case you want to alter the code.

NOTE: all these packages/dependencies can be downloaded using pip


## Usage
As mentioned, this program can be used simply by anyone. If you just want to test out the CNN model by providing it with
some dog images, you only need to use the **user.py** file.  
Firstly, you need to upload the images you want to test with into the folder **'images_to_be_identified_user'**.  
Ensure that the images are in a normal image format, if not, the code will terminate and give you a message telling
that the input image was incorrect.  
After you have uploaded the images to the folder, you can go to the **user.py** file and simply run it. You will see the
output telling you what the breed of the dog is.


## Requirements
In order to ensure that the code runs smoothly on all devices, I have added a gpu selector in the main file. Apart 
from this, all the images will be converted to greyscale and resized to dimensions 256x256. This prevents any memory or
runtime issues.  
If you just want to test out the CNN model, you don't need to have much software requirements. Any basic device that can
run python and its IDE will suffice.
However, if you wish to modify the code to alter the model, you will need a decent enough laptop that can handle all the
images. In case of any memory issues, the program will terminate and you will get a 137 error message. In that case you 
can just reduce the image dimensions or reduce the data set.


## Acknowledgments
I would first like to thank the Standford Dogs Dataset (http://vision.stanford.edu/aditya86/ImageNetDogs/) for providing
the data that was used in this project.

A huge thanks to numpy, opencv-python, tensorflow, os, scipy and matplotlib for providing their packages because of
which this project was possible in the first place.

I worked on this project alone and will not be actively working on this project anymore (I will be creating other 
related projects). However, I would love any suggestions/feedback/collaborative requests.


## Author and Date
by Jai Joshi  
Uploaded on 27th January, 2024
