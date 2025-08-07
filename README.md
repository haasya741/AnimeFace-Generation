Introduction: 
Generative Adversarial Networks (GANs) have revolutionized the field of generative modeling by creating realistic synthetic data from random noise. 
Among various types of GANs, the Deep Convolutional GAN (DCGAN) has proven to be particularly effective for generating high-quality images. 
This project implements a DCGAN for generating anime faces, using a simple architecture and exploring several variations to optimize image quality. 
The goal is to synthesize visually appealing anime faces by training a generator to mimic real data distributions and a discriminator to distinguish between real and fake images.


<img width="940" height="460" alt="image" src="https://github.com/user-attachments/assets/cbed6bd8-cbc0-4112-9e33-92cbcbb53e09" />


Dataset:
The dataset is taken from Kaggle over here(https://www.kaggle.com/datasets/soumikrakshit/anime-faces?resource=download). 
The data was obtained from www.getchu.com and processed using a face detector based on the repo (https://github.com/nagadomi/lbpcascade_animeface.)
The dataset contains images of size 64 by 64 pixels.


ARCHITECTURE:

Generator:
<img width="492" height="658" alt="image" src="https://github.com/user-attachments/assets/853804f5-609b-409f-8c4f-8496eb4d16ad" />

Discriminator:
<img width="913" height="1189" alt="image" src="https://github.com/user-attachments/assets/ed4bc6e8-7eb4-4e02-8db7-9d49504a2d83" />


DCGAN MODEL:
<img width="940" height="732" alt="image" src="https://github.com/user-attachments/assets/929b11da-adf6-4cbe-a7ad-9ae13d01d327" />

OUTPUT SAMPLE:

<img width="543" height="484" alt="image" src="https://github.com/user-attachments/assets/473bab24-e69b-4c60-ac09-a75129345ad2" />




