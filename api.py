
#importing Modules
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import tensorflow as tf
from PIL import Image
import numpy as np
import flask
import base64
import io   
import sys

app= flask.Flask(__name__)
model = None

#Load Model
def load_model():
    #loading model
    global model
    model = ResNet50(weights="imagenet")

    #loading Model Graph
    global graph
    graph = tf.get_default_graph()

def prepare_image(image,target):
    #image Preprocessing Module

    #Check if Image is in RGB or not if not then convert it using PIL
    if(image.mode != "RGB"):
        image=image.convert("RGB")

    #image preprocessing to make sure that the imput image size matches the input tensor size
    image = image.resize(target)
    #converting image to an image array using PIL library
    image = img_to_array(image)
    #Making the image suitable for tensor input
    image = np.expand_dims(image,axis=0)
    #preprocessing the image to make it ready for input to the model
    image = imagenet_utils.preprocess_input(image)

    return image

@app.route("/api",methods=["POST"])
def predict():

    data = {"success":False}
    #Check that image is uploaded to the server
    if(flask.request.method == "POST"):
        if(flask.request.files.get("image")):
            #debug Flag
            print("Entered into Core ML Engine",file=sys.stdout)
            #Read the image using PIL 
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            #Process image
            image = prepare_image(image, target=(224, 224))
            #use tf graph to predict the output
            with graph.as_default():
                preds = model.predict(image)

            #Storing the predicted classes into list
            results = imagenet_utils.decode_predictions(preds)

            data["predictions"] = []
            #iterate over the result to find predicted output 
            for(imagenetID,label,prob) in results[0]:
                r = {"label":label,"probability":float(prob)}
                data["predictions"].append(r)

            #This flag indicates that process was successful 
            data["success"]=True

    #return Json Data to the users
    return flask.jsonify(data)


@app.route("/")
def home():
    return flask.render_template("index.html")


if __name__ =="__main__":
    print("Its working we are making the systems Ready")
    load_model()
    app.run()