# to get the predict of product selection and run the coppeliasim software simulator to pick the product by cobot.
import sim  # copppeliasim  api package
import sys # coppeliasim api package
import time
import numpy as np # NumPy is the fundamental package for scientific computing in Python.It can be used to perform a wide variety of mathematical operations on arrays.
from simConst import * # coppeliasim api package
from PIL import Image #This library provides extensive file format support, an efficient internal representation, and fairly powerful image processing capabilities.
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image


def get_image_from_simulator(clientID,cam_handle):
    iter=60
    for i in range(iter): # repeat iter number of times
        time.sleep(1)
        # Check if there is an object under the camera:
        _,signal=sim.simxGetIntegerSignal(clientID,"captureProductPicture",sim.simx_opmode_blocking)
        
        if int(signal)==1:
            print("Image has been recieved by the signal of proximity sensor")
            # to identify a object under the camera
            err, resolution, image = sim.simxGetVisionSensorImage(clientID, cam_handle, 0, sim.simx_opmode_blocking)
            img = np.array(image,dtype=np.uint8)
            print(img.shape,"shape") # get the shape of the product
            img.resize([resolution[1],resolution[0],3])
            # take the current image and save in the file for prediction
            if err == sim.simx_return_ok:
                im = Image.fromarray(img)
                im.save("current_image.png")
    
            elif err == sim.simx_return_novalue_flag:
                print ("no image yet")
                pass
            else:
              print (err)
            return image
          
# for predication the current image of class.
def predict_class(model,img_width, img_height):
    img_str="current_image.png"
    # PREDICT THE CLASS OF ONE IMAGE
    img = image.load_img(img_str, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes_predic = model.predict(images) # predict the image of model
    classes_predic = np.argmax(classes_predic, axis=1)
    class_names=("bottle","fruit",)
    print ("predicted the class",class_names[classes_predic[0]],"Class code:", classes_predic[0],)
    return classes_predic[0],class_names[classes_predic[0]]

def load_saved_model(model_name):
    model=load_model(model_name)
    print(model.summary())
    return model

def pick_and_classify(clientID,class_code):
    print("Sent command for picking the product")
    inputInts=[class_code]
    inputFloats=[0.0]
    inputStrings=["class_name"]
    inputBuffer=bytearray()
    inputBuffer.append(78)
    res,retInts,retFloats,retStrings,retBuffer=sim.simxCallScriptFunction(clientID,'Franka',sim.sim_scripttype_childscript,
               'take_for_classification',inputInts,inputFloats,inputStrings,inputBuffer,simx_opmode_blocking)
    time.sleep(3)
    print("finished sending pick up command to cobot")
    return(res)


def main():
    sim.simxFinish(-1)
    # connect the remote api server with coppeliasim software
    clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5)
    # check the connection
    if clientID!=-1:
        print ("Connected to remote API server")
    else:
        print("Not connected to remote API server")
        sys.exit("Could not connect")

    _=sim.simxStartSimulation(clientID,sim.simx_opmode_blocking)

    res, v1 = sim.simxGetObjectHandle(clientID, 'Vision_sensor_camera', sim.simx_opmode_blocking) # get the object data of vision sensor camera

    img_width=64
    img_height=64
    model=load_saved_model("image_classification.h5") # load the training data of image classification.
    num_repeat=5 # the number of times out model will run

    for i in range(num_repeat):
        get_image_from_simulator(clientID,v1)
        class_code,class_name=predict_class(model,img_width, img_height)
        pick_and_classify(clientID,class_code)

    # wait 10 seconds before stop the simulation
    time.sleep(10)

    _=sim.simxStopSimulation(clientID, simx_opmode_blocking) # stop the simulator function

if __name__ == '__main__':
    main()

