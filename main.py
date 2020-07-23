  
import cv2, os, time, platform
import numpy as np
from argparse import ArgumentParser
from src.input_feeder import InputFeeder
from src.mouse_controller import MouseController
from src.face_detection import Model_Face_Detection
from src.facial_landmarks_detection import Model_Facial_Landmarks_Detection
from src.head_pose_estimation import Model_Head_Pose_Estimation
from src.gaze_estimation import Model_Gaze_Estimation

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
        inputFile = 0
    
    #check if the input file exists
    if (inputFile!=0) and (not os.path.isfile(inputFile)):
        print('Could not fine the input file : ', inputFile)
        exit(1)
    elif DEBUG:
        print('found input file : ', inputFile)

    #initialize the models
    fdmodel = Model_Face_Detection(model_path=faceDetectionModelPath, device=device, extensions=cpu_extension, prob_threshold=prob_threshold)
    fldmodel = Model_Facial_Landmarks_Detection(model_path=facialLandmarksDetectionmodelPath, device=device, extensions=cpu_extension, prob_threshold=prob_threshold)
    hpmodel = Model_Head_Pose_Estimation(model_path=headPoseModelPath, device=device, extensions=cpu_extension, prob_threshold=prob_threshold)
    gemodel = Model_Gaze_Estimation(model_path=modelPath, device=device, extensions=cpu_extension, prob_threshold=prob_threshold)

if __name__ == '__main__':
    main()