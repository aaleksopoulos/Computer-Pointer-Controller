  
import cv2, os, time, platform
import numpy as np
from argparse import ArgumentParser
from src.input_feeder import InputFeeder
from src.mouse_controller import MouseController
from src.face_detection import Model_Face_Detection
from src.facial_landmarks_detection import Model_Facial_Landmarks_Detection
from src.head_pose_estimation import Model_Head_Pose_Estimation
from src.gaze_estimation import Model_Gaze_Estimation
from src.mouse_controller import MouseController

DEBUG = True #helper function

def build_argparser():
    parser = ArgumentParser()

    parser.add_argument("-fd", "--faceDetectionModel", type=str, 
                        default='src/models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001', 
                        help="Path to the Face Detection model, without any extensions. Default to current project's file.")
    parser.add_argument("-fl", "--facialLandmarksDetectionmodel", type=str,
                        default='src/models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009', 
                        help="Facial Landmark Detection model, without any extensions. Default to current project's file - FP32 precision.")
    parser.add_argument("-hp", "--headPoseModel", type=str,
                        default='src/models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001', 
                        help="Path to the Head Pose Estimation model, without any extensions. Default to current project's file - FP32 precision.")
    parser.add_argument("-ge", "--gazeEstimationModel", type=str,
                        default='src/models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002', 
                        help="Path to the Gaze Estimation model, without any extensions. Default to current project's file - FP32 precision.")
    parser.add_argument("-i", "--input", type=str,
                        default='bin/demo.mp4',
                        help=" Path to video file or enter CAM to use the webcam. Default to the video provided by the instructors")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="path the CPU exntension file - not requirred for OpenVINO 2019R3 and later. Auto for OpenVINO to try and figure the extension by itself")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for the models.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to run inference on: CPU, GPU, FPGA, MYRIAD. Default: CPU")
    parser.add_argument("-sp", "--show_preview", default=True, type=bool,
                        help="Set to False if no preview video output is needed. Default: True")
    
    return parser

def get_cpu_extension():
    #NOTE Only applicable in the case of OPENVino version 2019R3 and lower
    #TODO check it on system with openvino < 2019R3 to check for the AVX type, although the SSE one can be used in AVX systems
    if (platform.system() == 'Windows'):
        CPU_EXTENSION = "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\\bin\intel64\Release\cpu_extension_avx2.dll"
    elif (platform.system() == 'Darwin'): #MAC
        CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
    else: #Linux, only the case of sse
        CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    return CPU_EXTENSION

def main():
    # Read input provided by the user
    args = build_argparser().parse_args()
    prob_threshold = args.prob_threshold
    device = args.device
    faceDetectionModelPath = args.faceDetectionModel
    facialLandmarksDetectionmodelPath = args.facialLandmarksDetectionmodel
    headPoseModelPath = args.headPoseModel
    gazeEstimationModelPath = args.gazeEstimationModel
    inputFile = args.input
    if args.cpu_extension == 'auto':
        cpu_extension = get_cpu_extension()
    else:
        cpu_extension = args.cpu_extension

    if DEBUG:
        print("probability threshold: ", prob_threshold)
        print("device: ", device)
        print("faceDetectionModelPath: ", faceDetectionModelPath)      
        print("facialLandmarksDetectionmodelPath: ", facialLandmarksDetectionmodelPath) 
        print("headPoseModelPath: ", headPoseModelPath) 
        print("gazeEstimationModelPath: ", gazeEstimationModelPath) 
        print('cpu extension : ', cpu_extension)
        print('input file or device : ', inputFile)

    #check if the models provided exist
    modelPaths = [faceDetectionModelPath, facialLandmarksDetectionmodelPath, headPoseModelPath, gazeEstimationModelPath]
    for modelPath in modelPaths:
        if not os.path.isfile(modelPath+'.xml'):
            print('Could not find the ' + modelPath+'.xml file')
            exit(1)
        if not os.path.isfile(modelPath+'.bin'):
            print('Could not find the ' + modelPath+'.bin file')
            exit(1)
        if DEBUG:
            print('Path to : ', modelPath, " exists.")

    if inputFile.lower()=='cam':
        inputFeeder = InputFeeder("cam")
    else:
        if not os.path.isfile(inputFile):
            print('Could not fine the input file : ', inputFile)
            exit(1)
        elif DEBUG:
            print('found input file : ', inputFile)
        inputFeeder = InputFeeder(input_type="video", input_file=inputFile)

    startTime = time.time()

    #initialize the models
    fdmodel = Model_Face_Detection(model_path=faceDetectionModelPath, device=device, extensions=cpu_extension, prob_threshold=prob_threshold)
    fldmodel = Model_Facial_Landmarks_Detection(model_path=facialLandmarksDetectionmodelPath, device=device, extensions=cpu_extension, prob_threshold=prob_threshold)
    hpmodel = Model_Head_Pose_Estimation(model_path=headPoseModelPath, device=device, extensions=cpu_extension, prob_threshold=prob_threshold)
    gemodel = Model_Gaze_Estimation(model_path=modelPath, device=device, extensions=cpu_extension, prob_threshold=prob_threshold)

    #load the models
    fdmodel.load_model()
    fldmodel.load_model()
    hpmodel.load_model()
    gemodel.load_model()

    modelLoadTime = time.time() - startTime

    print('Time to load the models: {:8.5f} seconds.'.format(modelLoadTime))

    inputFeeder.load_data()
    mouseController = MouseController('medium','fast')

    counter = 0
    inferenceTime = 0


    width, height = inputFeeder.get_size()
    print("width: ", width, " ,height: ", height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_capt = cv2.VideoWriter('output_video.mp4', fourcc, 25, (width,height))

    for  frame in inputFeeder.next_batch():

        if frame is None:
            break #finished the loop
        else:
            #to cancel easily
            key_pressed = cv2.waitKey(60)
            if key_pressed == 27:
                break 
            counter+=1
            startInference = time.time() #start the inference
            print('been here')
            #get the face and its coordinates from the face detection model
            faceCoords, faceImg = fdmodel.predict(frame)
            #cv2.imshow('face image', faceImg)
            #cv2.waitKey(0)
            print('got out of fdmodel')
            #get the eye positions and the bounding boxes
            left_eye, right_eye, eye_box_coords = fldmodel.predict(faceImg)
            print('got out of fldmodel')
            #get the angle of the head pose
            head_pose_angle = hpmodel.predict(faceImg)

            #get the gaze vector
            mouse_coords, gaze_vector = gemodel.predict(left_eye, right_eye, head_pose_angle)

            inferenceTime += time.time()-startInference #done with inferencing

            #update the cursor position every 5 frames
            if counter%5 == 0:
                print('been here')
                mouseController.move(mouse_coords[0], mouse_coords[1])
                cv2.imshow('video', cv2.resize(frame, (500, 500)))

            if args.show_preview and False:
                #show the face
                print('facecoords: ', faceCoords)
                cv2.rectangle(frame, (faceCoords[0], faceCoords[1]), (faceCoords[2], faceCoords[3]), (255, 255, 0), 2)
                #show the eyes
                cv2.rectangle(frame, (faceCoords[0]+eye_box_coords[0][0], faceCoords[1]+eye_box_coords[0][1]), (faceCoords[0]+eye_box_coords[0][2], faceCoords[1]+eye_box_coords[0][3]), (255, 255, 0), 2)
                cv2.rectangle(frame, (faceCoords[0]+eye_box_coords[1][0], faceCoords[1]+eye_box_coords[1][1]), (faceCoords[0]+eye_box_coords[1][2], faceCoords[1]+eye_box_coords[1][3]), (255, 255, 0), 2)
                #write out the gaze vector
                cv2.putText(frame, "Gaze Cordss: yaw= {:.2f} , pitch= {:.2f} , roll= {:.2f}".format(gaze_vector[0], gaze_vector[1], gaze_vector[2]), (20, 40), cv2.FONT_HERSHEY_COMPLEX,1, (255, 255, 0), 2)


            vid_capt.write(frame)
if __name__ == '__main__':
    main()