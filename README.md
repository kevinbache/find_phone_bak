## A Few Notes
* The basic idea is to treat this as something close to an image segmentation problem in which pixels in the phone are foreground and other pixels are background.  
* Since we don't have ground truth segmentation maps and I didn't want to hand label the training images, I just created artificial label images where there's a Gaussian centered on each phone's gound truth location.  The label values can be thought of as something like "probability that this pixel is in the phone area".
* For the model, I'm using an implementation of Deeplab V3 which I found at https://github.com/bonlime/keras-deeplab-v3-plus.
* Since we have soft label images, this is really more of a pixel-wise softmax regression rather than a true image segmentation problem but the difference is more semantic than architectural.
* One epoch of training seems to do it but I dropped in 3 just for good measure.  
* The requirements.txt file is just a `pip freeze` call from the default Google Cloud deep learning image I used for training.  The main dependencies are Keras, TensorFlow, Scikit Image, Pandas, and Numpy.
* I didn't actually install