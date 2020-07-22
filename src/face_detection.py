import os, cv2
import numpy as np
from openvino.inference_engine import IECore

DEBUG = False

class Model_Face_Detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, prob_threshold=0.6):
        '''
        TODO: Use this to set your instance variables.
        '''
        #raise NotImplementedError
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_structure = model_name+'.xml'
        self.model_weights = model_name+'.bin'
        self.core = None
        self.network = None
        self.exec_net = None
        self.input_blob = None
        self.output_blob = None
        self.input_shape = None
        self.output_shape = None
        self.prob_threshold = prob_threshold

    def get_unsupported_layers(self):
        '''
        Returns a list of the unsupported layers
        NOTE For OpenVINO version 2020 and above, the cpu_extension is not needed
        '''
        #get a list of the supported layers
        supported_layers = self.core.query_network(self.network, device_name=self.device)
        #get the required layers
        required_layers = list(self.network.layers.keys())
        #check if there are unsupported layers
        unsupported_layers = []
        for layer in required_layers:
            if layer not in supported_layers:
                unsupported_layers.append(layer)

        return unsupported_layers

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        #raise NotImplementedError
        #check if we have provided a valid file for the model files
        if not self.check_model():
            exit(1)

        #initialize the Inference Engine and get the instance of executable network
        self.core = IECore()
        self.network = self.core.read_network(model=self.model_structure, weights=self.model_weights)
        self.exec_net = self.core.load_network(network=self.network, device_name=self.device, num_requests=1)

        #check if there are any unsupported layers
        unsupported_layers = self.get_unsupported_layers()

        #if there are any unsupported layers, add CPU extension, if avaiable
        if (len(unsupported_layers)>0) and (self.device=='CPU'):
            print("There are unsupported layers found, will try to add CPU extension...")
            self.core.add_extension(extension_path=self.extensions, device=self.device)

        #add, if provided, a cpu extension
        if (self.extensions):
            self.core.add_extension(self.extensions)

        #recheck for unsupported layers, and exit if there are any
        unsupported_layers = self.get_unsupported_layers()
        if (len(unsupported_layers)>0):
            print("After adding CPU extension, there are still unsupported layers, exiting...")
            exit(1)
        
        #load to network to get the executable network
        self.exec_network = self.core.load_network(self.network, self.device)

        #get the input and output blobs
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        #get the shape of the input and output
        self.input_shape = self.network.inputs[self.input_blob].shape
        self.output_shape = self.network.outputs[self.output_blob].shape

        return self.exec_network

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        #raise NotImplementedError
        prep_img = self.preprocess_input(image)
        output_frame = self.exec_net.infer({self.input_blob : prep_img})
        tracked_list, ret_img = self.preprocess_output(output_frame, image)
        
        return tracked_list, ret_img


    def check_model(self):
        '''
        If the path to the model xml and bin files exists, returns True, else False
        '''
        #raise NotImplementedError
        if ((os.path.exists(self.model_structure)) and (os.path.exists(self.model_weights))):
            if DEBUG:
                print("model found")
                print("model_xml: ", self.model_structure)
                print("model_bin: ", self.model_weights)
                print("device: ", self.device)
            return True
        else:
            print("There was a problem reading the xml file provided, exiting...")
            return False

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''     
        #raise NotImplementedError
        height = self.input_shape[2]
        width = self.input_shape[3]
        img = cv2.resize(image, (width, height))
        img = img.transpose((2,0,1))
        return img.reshape(1, 3, width, height)

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        #raise NotImplementedError
        #get the image width and height
        height = image.shape[0]
        width = image.shape[1]

        tracked_list = [] #to keep track what the model tracked
        ret_img = image #the image to return

        for fr in outputs:
            if (fr[0][0][0] == -1): #if we have not detected anything, we break out
                break
            if (fr[0][0][2]>=self.prob_threshold): #if the probability is above the one stated
                
                x1 = int(fr[0][0][3]*width)
                y1 = int(fr[0][0][4]*height)
                x2 = int(fr[0][0][5]*width)
                y2 = int(fr[0][0][6]*height)
                if DEBUG:
                    print("--------------------------")
                    print("calucalated x1: ", x1)
                    print("calucalated x2: ", x2)
                    print("calucalated y1: ", y1)
                    print("calucalated y2: ", y2)
                    print("--------------------------")
                tracked_list.append([x1, y1, x2, y2])
                ret_img = image[y1:y2, x1:x2]
        
        return tracked_list, ret_img
