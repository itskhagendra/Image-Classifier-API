# Image Classifier API Using Keras And Flask

This project contains the code for creating a REST API for a Image classifictaion model using a pretrained model in this case I am using Resnet-50 but you can use any model you want.

just make sure that you change the relevent paramets if you change the models such as tensor size and weights 

`model = ResNet50(weights="imagenet")`

`from keras.applications import ResNet50`

`image = prepare_image(image, target=(224, 224))`

##