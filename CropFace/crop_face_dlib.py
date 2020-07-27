import dlib
import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage import transform as trans
from skimage import io


detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
sp = dlib.shape_predictor('./shape_predictor_5_face_landmarks.dat')
def get_template_from_files(data_root, save_template, use_examples=100):
    predictor_path = './shape_predictor_5_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    points = [] 
    for idx, name in tqdm(enumerate(sorted(os.listdir(data_root))), total=use_examples):
        img = dlib.load_rgb_image(os.path.join(data_root, name))
        dets = detector(img, 1) 
        if len(dets) > 0:
            shape = sp(img, dets[0])
            single_points = []
            for i in range(5):
                single_points.append([shape.part(i).x, shape.part(i).y])
            points.append(single_points)
        if idx - 1 >= use_examples: break
    points = np.array(points)
    template = np.mean(points, axis=0)
    np.save(save_template, template)
    return template

def get_5_points(img):
    dets = detector(img, 1)
    if len(dets) == 0:
        return None
    shape = sp(img, dets[0].rect) 
    single_points = []

    for i in range(5):
        single_points.append([shape.part(i).x, shape.part(i).y])
    return np.array(single_points) 

def align_and_save(img_path, save_path, template_scale=1):
    out_size = (512, 512) 
    img = dlib.load_rgb_image(img_path)
    source = get_5_points(img) 
    print(source)
    if source is None: #
        print('No face detect')
        return
    tform = trans.SimilarityTransform()                                                                                                                                                  
    tform.estimate(source, reference)
    M = tform.params[0:2,:]
    crop_img = cv2.warpAffine(img, M, out_size)
    io.imsave(save_path, crop_img)    
    print('Saving image', img_path)


######parameters for test
datasets_path = './TestWhole/WholeImgs' #
Save_path = '../TestData/RealVgg/Imgs' ###obtain the landmarks

reference = np.load('./FFHQ_template.npy') / 2 #

if not os.path.exists(Save_path):
    os.mkdir(Save_path)
ImgNames = os.listdir(datasets_path)
ImgNames.sort()
for i, ImgName in enumerate(ImgNames):
    print(i)
    ImgPath = os.path.join(datasets_path,ImgName)
    SavePath = os.path.join(Save_path,ImgName)
    align_and_save(ImgPath, SavePath)
