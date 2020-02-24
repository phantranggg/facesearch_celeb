from config import get_config
from Learner import face_learner
from PIL import Image
import glob
import torch
from utils import load_facebank, draw_box_name, prepare_facebank
from RetinaFace_Pytorch import torchvision_model, eval_widerface
import datetime
import numpy as np
import time, re, os, glob, cv2, argparse
import pickle5 as pickle
from shutil import copyfile

from _3ddfa import mobilenet_v1
import torchvision.transforms as transforms
from _3ddfa.utils.ddfa import ToTensorGjz, NormalizeGjz
from _3ddfa.utils.inference import crop_img, predict_68pts
from _3ddfa.utils.estimate_pose import parse_pose

def dump(emb_list, filename_list):
    emb_list = np.asarray(emb_list)
    with open('emb_list.pkl', 'wb') as f:
        pickle.dump(emb_list, f)
    with open('filename_list.pkl', 'wb') as f:
        pickle.dump(filename_list, f)

def load_pretrain_model():
    checkpoint_fp = './_3ddfa/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    model.eval()
    return model

def cal_frontal_score(pose_angle):
    return pose_angle[0]**2 + pose_angle[1]**2 + pose_angle[2]**2

def run_3ddfa(dataset_dir, save_dir):
    filename_list = []
    emb_list = []
    num_imgs = 0
    best_frontal_img, best_frontal_filename = None, None
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    model_3ddfa = load_pretrain_model()

    # for filename in glob.glob(dataset_dir + "*.jpg"):
    for subdir, dirs, files in os.walk(DB_DIR):
        MIN_SCORE = 3 * 180**2
        best_frontal_filename = None
        best_frontal_img = None
        for filename in files:
            # print(subdir, filename)

            num_imgs += 1
            if num_imgs % 10000 == 0:
                print(num_imgs)
                dump(emb_list, filename_list)
            
            # crop + align input image
            img = Image.open(os.path.join(subdir, filename))
            bboxes, faces = eval_widerface.align_multi(img, RetinaFace, 1, 16)
            if bboxes is None or bboxes.shape != torch.Size([1, 4]):
                continue 
            bboxes = bboxes.cpu().detach().numpy().squeeze(0)
            # print(bboxes)

            # get pose list (pitch, roll, yaw)
            face_BGR = np.array(faces[0])  
            face_BGR = face_BGR[:, :, ::-1].copy() # Convert RGB to BGR
            input_face = transform(face_BGR).unsqueeze(0)
            with torch.no_grad():
                param = model_3ddfa(input_face)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            pts68 = predict_68pts(param, bboxes)
            P, pose, pose_angle = parse_pose(param)

            # get best pose of a celebrity
            frontal_score = cal_frontal_score(pose_angle)
            if frontal_score < MIN_SCORE:
                MIN_SCORE = frontal_score
                best_frontal_img = faces[0]
                best_frontal_filename = filename
            print(subdir, filename, frontal_score, MIN_SCORE, best_frontal_filename)

        if best_frontal_filename is None or best_frontal_img is None:
            continue

        # embed face
        emb = learner.embedding(conf, best_frontal_img, True)
        emb = emb.cpu().detach().numpy().squeeze(0)

        # if num_imgs % 10000 == 0:
            # print(num_imgs)
            # dump(emb_list, filename_list)x    

        copyfile(os.path.join(subdir, best_frontal_filename), save_dir+"{}.jpg".format(num_imgs))
        # filename = re.split(r'/', os.path.join(subdir, filename))[-2:]
        filename_list.append(save_dir+"{}.jpg".format(num_imgs))
        emb_list.append(emb)

    return emb_list, filename_list, num_imgs

def run(dataset_dir):
    filename_list = []
    emb_list = []
    num_imgs = 0

    for filename in glob.glob(dataset_dir + "*.jpg"):
        num_imgs += 1
        # print(filename)
        img = Image.open(filename)
        bboxes, faces = eval_widerface.align_multi(img, RetinaFace, 1, 16)
        if len(faces) == 0:
            continue
        emb = learner.embedding(conf, faces[0], True)
        emb = emb.cpu().detach().numpy().squeeze(0)

        # filename = re.split(r'/', os.path.join(subdir, filename))[-2:]
        filename_list.append(filename)
        emb_list.append(emb)

        if num_imgs % 10000 == 0:
            print(num_imgs)
            dump(emb_list, filename_list)

    return emb_list, filename_list, num_imgs

if __name__ == '__main__':
    conf = get_config(False)

    # Create torchvision model
    return_layers = {'layer2': 1,'layer3': 2,'layer4': 3}
    RetinaFace = torchvision_model.create_retinaface(return_layers)

    # Load trained model
    retina_dict = RetinaFace.state_dict()
    pre_state_dict = torch.load('./RetinaFace_Pytorch/model.pt')
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
    RetinaFace.load_state_dict(pretrained_dict)

    RetinaFace = RetinaFace.cuda()
    RetinaFace.eval()
    print('Retinaface loaded')
    
    print('loading learner')
    learner = face_learner(conf, True)
    learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')

    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir", default="./data/test_1")
    parser.add_argument("--save_dir", default="./data/test_3dffa/")
    args = parser.parse_args()

    # DB_DIR = '../celeb'
    DB_DIR = args.db_dir
    SAVE_DIR = args.save_dir

    # start = time.time()
    # emb_list, filename_list, num_imgs = run(DB_DIR)
    emb_list, filename_list, num_imgs = run_3ddfa(DB_DIR, SAVE_DIR)
    dump(emb_list, filename_list)
    # print('Crop image + emb time: ', time.time() - start)
