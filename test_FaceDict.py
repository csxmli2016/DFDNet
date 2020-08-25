import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_crop
from util import html
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
import torch
import random
import cv2
import dlib
from skimage import transform as trans
from skimage import io
from data.image_folder import make_dataset
import sys
sys.path.append('FaceLandmarkDetection')
import face_alignment

###########################################################################
################# functions of crop and align face images #################
###########################################################################
def get_5_points(img):
    dets = detector(img, 1)
    if len(dets) == 0:
        return None
    areas = []
    if len(dets) > 1:
        print('\t###### Warning: more than one face is detected. In this version, we only handle the largest one.')
    for i in range(len(dets)):
        area = (dets[i].rect.right()-dets[i].rect.left())*(dets[i].rect.bottom()-dets[i].rect.top())
        areas.append(area)
    ins = areas.index(max(areas))
    shape = sp(img, dets[ins].rect) 
    single_points = []
    for i in range(5):
        single_points.append([shape.part(i).x, shape.part(i).y])
    return np.array(single_points) 

def align_and_save(img_path, save_path, save_input_path, save_param_path, upsample_scale=2):
    out_size = (512, 512) 
    img = dlib.load_rgb_image(img_path)
    h,w,_ = img.shape
    source = get_5_points(img) 
    if source is None: #
        print('\t################ No face is detected')
        return
    tform = trans.SimilarityTransform()                                                                                                                                                  
    tform.estimate(source, reference)
    M = tform.params[0:2,:]
    crop_img = cv2.warpAffine(img, M, out_size)
    io.imsave(save_path, crop_img) #save the crop and align face
    io.imsave(save_input_path, img) #save the whole input image
    tform2 = trans.SimilarityTransform()  
    tform2.estimate(reference, source*upsample_scale)
    # inv_M = cv2.invertAffineTransform(M)
    np.savetxt(save_param_path, tform2.params[0:2,:],fmt='%.3f') #save the inverse affine parameters
    
def reverse_align(input_path, face_path, param_path, save_path, upsample_scale=2):
    out_size = (512, 512) 
    input_img = dlib.load_rgb_image(input_path)
    h,w,_ = input_img.shape
    face512 = dlib.load_rgb_image(face_path)
    inv_M = np.loadtxt(param_path)
    inv_crop_img = cv2.warpAffine(face512, inv_M, (w*upsample_scale,h*upsample_scale))
    mask = np.ones((512, 512, 3), dtype=np.float32) #* 255
    inv_mask = cv2.warpAffine(mask, inv_M, (w*upsample_scale,h*upsample_scale))
    upsample_img = cv2.resize(input_img, (w*upsample_scale, h*upsample_scale))
    inv_mask_erosion_removeborder = cv2.erode(inv_mask, np.ones((2 * upsample_scale, 2 * upsample_scale), np.uint8))# to remove the black border
    inv_crop_img_removeborder = inv_mask_erosion_removeborder * inv_crop_img
    total_face_area = np.sum(inv_mask_erosion_removeborder)//3
    w_edge = int(total_face_area ** 0.5) // 20 #compute the fusion edge based on the area of face
    erosion_radius = w_edge * 2
    inv_mask_center = cv2.erode(inv_mask_erosion_removeborder, np.ones((erosion_radius, erosion_radius), np.uint8))
    blur_size = w_edge * 2
    inv_soft_mask = cv2.GaussianBlur(inv_mask_center,(blur_size + 1, blur_size + 1),0)
    merge_img = inv_soft_mask * inv_crop_img_removeborder + (1 - inv_soft_mask) * upsample_img
    io.imsave(save_path, merge_img.astype(np.uint8))

###########################################################################
################ functions of preparing the test images ###################
###########################################################################
def AddUpSample(img):
    return img.resize((512, 512), Image.BICUBIC)
def get_part_location(partpath, imgname):
    Landmarks = []
    if not os.path.exists(os.path.join(partpath,imgname+'.txt')):
        print(os.path.join(partpath,imgname+'.txt'))
        print('\t################ No landmark file')
        return 0
    with open(os.path.join(partpath,imgname+'.txt'),'r') as f:
        for line in f:
            tmp = [np.float(i) for i in line.split(' ') if i != '\n']
            Landmarks.append(tmp)
    Landmarks = np.array(Landmarks) 
    Map_LE = list(np.hstack((range(17,22), range(36,42))))
    Map_RE = list(np.hstack((range(22,27), range(42,48))))
    Map_NO = list(range(29,36))
    Map_MO = list(range(48,68))
    try:
        #left eye
        Mean_LE = np.mean(Landmarks[Map_LE],0)
        L_LE = np.max((np.max(np.max(Landmarks[Map_LE],0) - np.min(Landmarks[Map_LE],0))/2,16))
        Location_LE = np.hstack((Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
        #right eye
        Mean_RE = np.mean(Landmarks[Map_RE],0)
        L_RE = np.max((np.max(np.max(Landmarks[Map_RE],0) - np.min(Landmarks[Map_RE],0))/2,16))
        Location_RE = np.hstack((Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)
        #nose
        Mean_NO = np.mean(Landmarks[Map_NO],0)
        L_NO = np.max((np.max(np.max(Landmarks[Map_NO],0) - np.min(Landmarks[Map_NO],0))/2,16))
        Location_NO = np.hstack((Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)
        #mouth
        Mean_MO = np.mean(Landmarks[Map_MO],0)
        L_MO = np.max((np.max(np.max(Landmarks[Map_MO],0) - np.min(Landmarks[Map_MO],0))/2,16))
        Location_MO = np.hstack((Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)
    except:
        return 0
    return torch.from_numpy(Location_LE).unsqueeze(0), torch.from_numpy(Location_RE).unsqueeze(0), torch.from_numpy(Location_NO).unsqueeze(0), torch.from_numpy(Location_MO).unsqueeze(0)

def obtain_inputs(img_path, Landmark_path, img_name):
    A_paths = os.path.join(img_path,img_name)
    A = Image.open(A_paths).convert('RGB')
    Part_locations = get_part_location(Landmark_path, img_name)
    if Part_locations == 0:
        return 0
    C = A
    A = AddUpSample(A)
    A = transforms.ToTensor()(A) 
    C = transforms.ToTensor()(C)
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A) #
    C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C) #
    return {'A':A.unsqueeze(0), 'C':C.unsqueeze(0), 'A_paths': A_paths,'Part_locations': Part_locations}
    
if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.which_epoch = 'latest' #

    #######################################################################
    ########################### Test Param ################################
    #######################################################################
    # opt.gpu_ids = [0] # gpu id. if use cpu, set opt.gpu_ids = []
    # TestImgPath = './TestData/TestWhole' # test image path
    # ResultsDir = './Results/TestWholeResults' #save path 
    # UpScaleWhole = 4  # the upsamle scale. It should be noted that our face results are fixed to 512.
    TestImgPath = opt.test_path
    ResultsDir = opt.results_dir
    UpScaleWhole = opt.upscale_factor

    print('\n###################### Now Running the X {} task ##############################'.format(UpScaleWhole))

    #######################################################################
    ###########Step 1: Crop and Align Face from the whole Image ###########
    #######################################################################
    print('\n###############################################################################')
    print('####################### Step 1: Crop and Align Face ###########################')
    print('###############################################################################\n')
    
    detector = dlib.cnn_face_detection_model_v1('./packages/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('./packages/shape_predictor_5_face_landmarks.dat')
    reference = np.load('./packages/FFHQ_template.npy') / 2
    SaveInputPath = os.path.join(ResultsDir,'Step0_Input')
    if not os.path.exists(SaveInputPath):
        os.makedirs(SaveInputPath)
    SaveCropPath = os.path.join(ResultsDir,'Step1_CropImg')
    if not os.path.exists(SaveCropPath):
        os.makedirs(SaveCropPath)

    SaveParamPath = os.path.join(ResultsDir,'Step1_AffineParam') #save the inverse affine parameters
    if not os.path.exists(SaveParamPath):
        os.makedirs(SaveParamPath)

    ImgPaths = make_dataset(TestImgPath)
    for i, ImgPath in enumerate(ImgPaths):
        ImgName = os.path.split(ImgPath)[-1]
        print('Crop and Align {} image'.format(ImgName))
        SavePath = os.path.join(SaveCropPath,ImgName)
        SaveInput = os.path.join(SaveInputPath,ImgName)
        SaveParam = os.path.join(SaveParamPath, ImgName+'.npy')
        align_and_save(ImgPath, SavePath, SaveInput, SaveParam, UpScaleWhole)

    #######################################################################
    ####### Step 2: Face Landmark Detection from the Cropped Image ########
    #######################################################################
    print('\n###############################################################################')
    print('####################### Step 2: Face Landmark Detection #######################')
    print('###############################################################################\n')
    
    SaveLandmarkPath = os.path.join(ResultsDir,'Step2_Landmarks')
    if len(opt.gpu_ids) > 0:
        dev = 'cuda:{}'.format(opt.gpu_ids[0])
    else:
        dev = 'cpu'
    FD = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,device=dev, flip_input=False)
    if not os.path.exists(SaveLandmarkPath):
        os.makedirs(SaveLandmarkPath)
    ImgPaths = make_dataset(SaveCropPath)
    for i,ImgPath in enumerate(ImgPaths):
        ImgName = os.path.split(ImgPath)[-1]
        print('Detecting {}'.format(ImgName))
        Img = io.imread(ImgPath)
        try:
            PredsAll = FD.get_landmarks(Img)
        except:
            print('\t################ Error in face detection, continue...')
            continue
        if PredsAll is None:
            print('\t################ No face, continue...')
            continue
        ins = 0
        if len(PredsAll)!=1:
            hights = []
            for l in PredsAll:
                hights.append(l[8,1] - l[19,1])
            ins = hights.index(max(hights))
            # print('\t################ Warning: Detected too many face, only handle the largest one...')
            # continue
        preds = PredsAll[ins]
        AddLength = np.sqrt(np.sum(np.power(preds[27][0:2]-preds[33][0:2],2)))
        SaveName = ImgName+'.txt'
        np.savetxt(os.path.join(SaveLandmarkPath,SaveName),preds[:,0:2],fmt='%.3f')

    #######################################################################
    ####################### Step 3: Face Restoration ######################
    #######################################################################

    print('\n###############################################################################')
    print('####################### Step 3: Face Restoration ##############################')
    print('###############################################################################\n')

    SaveRestorePath = os.path.join(ResultsDir,'Step3_RestoreCropFace')# Only Face Results
    if not os.path.exists(SaveRestorePath):
        os.makedirs(SaveRestorePath)
    model = create_model(opt)
    model.setup(opt)
    # test
    ImgPaths = make_dataset(SaveCropPath)
    total = 0
    for i, ImgPath in enumerate(ImgPaths):
        ImgName = os.path.split(ImgPath)[-1]
        print('Restoring {}'.format(ImgName))
        torch.cuda.empty_cache()
        data = obtain_inputs(SaveCropPath, SaveLandmarkPath, ImgName)
        if data == 0:
            print('\t################ Error in landmark file, continue...')
            continue #
        total = total + 1
        model.set_input(data)
        try:
            model.test()
            visuals = model.get_current_visuals()
            save_crop(visuals,os.path.join(SaveRestorePath,ImgName))
        except Exception as e:
            print('\t################ Error in enhancing this image: {}'.format(str(e)))
            print('\t################ continue...')
            continue

    #######################################################################
    ############ Step 4: Paste the Results to the Input Image #############
    #######################################################################
    
    print('\n###############################################################################')
    print('############### Step 4: Paste the Restored Face to the Input Image ############')
    print('###############################################################################\n')

    SaveFianlPath = os.path.join(ResultsDir,'Step4_FinalResults')
    if not os.path.exists(SaveFianlPath):
        os.makedirs(SaveFianlPath)
    ImgPaths = make_dataset(SaveRestorePath)
    for i,ImgPath in enumerate(ImgPaths):
        ImgName = os.path.split(ImgPath)[-1]
        print('Final Restoring {}'.format(ImgName))
        WholeInputPath = os.path.join(TestImgPath,ImgName)
        FaceResultPath = os.path.join(SaveRestorePath, ImgName)
        ParamPath = os.path.join(SaveParamPath, ImgName+'.npy')
        SaveWholePath = os.path.join(SaveFianlPath, ImgName)
        reverse_align(WholeInputPath, FaceResultPath, ParamPath, SaveWholePath, UpScaleWhole)

    print('\nAll results are saved in {} \n'.format(ResultsDir))
    
