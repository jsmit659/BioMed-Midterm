# BioMed-Midterm

### Hardware Used:

CPU: 11th Gen Intel(R) Core(TM) i7-11600H @ 2.90GHz\
GPU: NVIDIA RTX3050Ti 8Gb RAM Boost Clock: 1.78GHz\
Memory :40GB DDR4 RAM\

### Introduction to SRGANs
Super Resolution GAN (SRGAN) Links to an external site.is a deep learning architecture that uses a combination of GANs 
and convolutional neural networks (CNNs) to generate high-resolution images from low-resolution images. The idea behind S
RGAN is to train a generator network to create high-resolution images that are as close as possible to the real high-resolution 
images, and a discriminator network that is trained to distinguish between the generated high-resolution images and real high-resolution images. 
The training process involves feeding low-resolution images to the generator, which then generates a high-resolution image. The discriminator 
then evaluates the generated high-resolution image and provides feedback to the generator to improve the quality of the generated image.
The generator and discriminator networks are trained iteratively until the generated images are of sufficient quality. Super Resolution GAN 
has many practical applications, such as in medical imaging, satellite imagery, and video processing. It can help to enhance the quality of 
low-resolution images, making them more useful for analysis and decision-making.

### Train a binary classifier (called A) on the dataset using transfer learning (exactly like Assignment 1). The images should be downscaled to 128x128
This requirement was completed in assignment one and is contained within the Model A file. The results for Model A are shown below. 

![image](https://user-images.githubusercontent.com/113131600/227605525-c456dc24-76ee-4f05-9119-7367c35b1793.png)


### Next, train the SRGAN to generate 128x128 images. Each image of the training is downscaled to 32x32.
To do this, the original images must first be downsized for their original shapes to a standard 32x32x3 image size. To do this, a loop was created 
to downsize the images. This code is shown in the block below.

```python
import os
import cv2
import numpy as np
# Set up paths
data_dir = '/home/jsmith/Desktop/JohnSmith_AssignmentMidtermProject/Data/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
output_dir = '/home/jsmith/Desktop/JohnSmith_AssignmentMidtermProject/Data/Downsized_32x32x3'

# Define image size
img_size = (32, 32)

# Loop through train and test folders
for folder in [train_dir, test_dir]:

    # Loop through DME and DRUSEN subfolders
    for subfolder in ['DME', 'DRUSEN']:

        # Set up subfolder path
        subfolder_path = os.path.join(folder, subfolder)

        # Loop through images in subfolder
        for img_name in os.listdir(subfolder_path):

            # Read in image and resize to img_size
            img_path = os.path.join(subfolder_path, img_name)
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, img_size)

            # Save resized image to output directory
            output_path = os.path.join(output_dir, folder.split('/')[-1], subfolder, img_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img_resized)

```


### Show some examples of scaled images in JNB

Once them Images have been downsized to 32x32x3, the images are extremly small and almost indistinguishable. Below are some examples of these newly generated downsized images.

![DME-323904-3](https://user-images.githubusercontent.com/113131600/227603680-65620948-0156-465b-82b6-fbb10561e790.jpeg)
![DME-328450-1](https://user-images.githubusercontent.com/113131600/227603684-6133a1d7-9f22-4d54-8301-9968e0afa103.jpeg)
![DME-383435-2](https://user-images.githubusercontent.com/113131600/227603685-91ed7b84-1f37-41d3-999c-53d25678891d.jpeg)
![DME-403817-3](https://user-images.githubusercontent.com/113131600/227603686-1725c00a-069d-4d4e-a148-a65e9cc5fd77.jpeg)
![DME-405804-8](https://user-images.githubusercontent.com/113131600/227603687-296aa918-8f90-4a11-92ca-bda78024bf90.jpeg)
![DME-405804-14](https://user-images.githubusercontent.com/113131600/227603688-56c0b5fd-00ed-4cd4-8623-c9aae8b1871c.jpeg)
![DME-405804-18](https://user-images.githubusercontent.com/113131600/227603689-964c788a-7c99-4a45-8b13-46a2c9364ed5.jpeg)
![DME-408201-12](https://user-images.githubusercontent.com/113131600/227603690-6e6b3945-799b-473d-93b2-fa9252b82810.jpeg)
![DME-408201-14](https://user-images.githubusercontent.com/113131600/227603692-fa19a9a7-d392-48b4-8145-2a6d408547e5.jpeg)
![DME-408201-16](https://user-images.githubusercontent.com/113131600/227603693-b572c501-9ef9-4c27-a502-693b78e51daa.jpeg)
![DME-415074-2](https://user-images.githubusercontent.com/113131600/227603694-2cc48ec4-d982-4979-b096-2c4215243829.jpeg)
![DME-446851-5](https://user-images.githubusercontent.com/113131600/227603695-4730a469-1bd9-4ea9-a310-b856d005c76e.jpeg)
![DME-446851-9](https://user-images.githubusercontent.com/113131600/227603698-309fa3d0-8278-49c2-bf3a-26d164ef784e.jpeg)
![DME-449872-1](https://user-images.githubusercontent.com/113131600/227603699-c0eb3f4c-2561-4782-a04c-2f82698c305c.jpeg)
![DME-462675-10](https://user-images.githubusercontent.com/113131600/227603700-ceca6863-fda9-4987-972d-a2a4032e1e3c.jpeg)
![DME-462675-21](https://user-images.githubusercontent.com/113131600/227603701-6cdfc0d4-e016-4523-9277-eb5b4f0a6170.jpeg)
![DME-462675-30](https://user-images.githubusercontent.com/113131600/227603704-a50abebb-e271-44e3-ad70-860d54a63ea1.jpeg)
![DME-462675-33](https://user-images.githubusercontent.com/113131600/227603705-b6a5be01-f549-44fd-b4ea-f00af4452eb6.jpeg)
![DME-462675-45](https://user-images.githubusercontent.com/113131600/227603706-4fbfb556-0d49-49c9-a1b0-ac282571d2d1.jpeg)
![DME-462675-49](https://user-images.githubusercontent.com/113131600/227603707-285386e2-22c3-4aa9-8b56-dde99844ac29.jpeg)
![DME-462675-56](https://user-images.githubusercontent.com/113131600/227603708-5a2a02a2-a5aa-4bef-96c9-1a675e51253e.jpeg)
![DME-462675-60](https://user-images.githubusercontent.com/113131600/227603709-d681da0b-1213-4bc3-a547-4fe89cf38f5c.jpeg)
![DME-462675-67](https://user-images.githubusercontent.com/113131600/227603710-806fbd4f-43f0-488d-8d0f-49418437b8c5.jpeg)
![DME-465734-3](https://user-images.githubusercontent.com/113131600/227603711-b4b90b94-0e06-4fa3-a91b-ffc35961b599.jpeg)
![DME-488037-1](https://user-images.githubusercontent.com/113131600/227603713-6201f7b0-1bd3-46bd-b513-072b87e057c5.jpeg)
![DME-509061-2](https://user-images.githubusercontent.com/113131600/227603714-2feff083-cd69-4b66-88f6-328e0c9807f0.jpeg)


### Utilize the images generated by SRGAN in order to train a new model (called B)
This requirment is completed in the Model B file.

### Train the SRGAN for at least 150 epochs
This requirment is done in the SRGAN File


### Apply normalization and image transformation, and demonstrate some of the transformed samples


### Compare the performance of both models using different metrics such as F1, Accuracy, AUC
The final step was to compare the two models; A and B.
The output when rerunning the model with the newly generated images, Model B, is shown below.

![image](https://user-images.githubusercontent.com/113131600/227605778-6b5144f3-be4e-46d6-9b3c-88e8468280ee.png)

From a comparison of the two loss and accuracy curves, including the confussion matrix, Model B performed better than expected for generated images, but not as well as the original. 
There are many considerations when reasoning why there is a difference and why there is succh drastic overfitting within Model B. One reason, and the most applicable here is the number of training Epochs ran. 
The original requirment for this dataset with succh low pixel density images was 500 Epochs, but due to time and memory constraints, the number was reduced to 150. More Training would lead to better and mroe accurate results.


## Basic Steps to create and run an SRGAN to take 32x32x3 images to 128x128x3 then perform binary classification

1. Collect and preprocess your dataset of 32x32x3 images. This can include resizing the images to 128x128x3, normalizing pixel values, and splitting the dataset into training and testing sets.

2. Train the SRGAN model on the training set of 32x32x3 images to generate 128x128x3 images. SRGAN is a deep neural network that uses a combination of convolutional and deconvolutional layers, as well as residual blocks, to generate high-quality images.

3. Fine-tune the SRGAN model on the training set of 128x128x3 images. This step involves using a discriminator network to differentiate between real and generated images, and adjusting the weights of the SRGAN network based on the feedback from the discriminator.

4. Evaluate the performance of the SRGAN model on the test set of 128x128x3 images. This can include calculating metrics such as peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM) to measure the similarity between the generated and real images.

5. Train a binary classifier on the generated 128x128x3 images to perform binary classification. This can involve using a variety of machine learning algorithms, such as logistic regression, support vector machines (SVMs), or deep neural networks.

6. Evaluate the performance of the binary classifier on a separate test set of generated 128x128x3 images. This can include calculating metrics such as accuracy, precision, recall, and F1 score to measure the effectiveness of the classifier in distinguishing between the two classes.

7. Fine-tune the SRGAN model and binary classifier as necessary based on the evaluation results. This may involve adjusting hyperparameters, optimizing the network architecture, or retraining the model on additional data.
