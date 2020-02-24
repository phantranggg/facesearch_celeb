import cv2
from PIL import Image
import argparse
from pathlib import Path
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import torchvision_model
import datetime
import eval_widerface
import numpy as np
import time
# import skvideo.io

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-f", "--file_name", help="video file name",default='video.mp4', type=str)
    parser.add_argument("-s", "--save_name", help="output file name",default='recording', type=str)
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    parser.add_argument("-b", "--begin", help="from when to start detection(in seconds)", default=0, type=int)
    parser.add_argument("-d", "--duration", help="perform detection for how long(in seconds)", default=0, type=int)

    # parser.add_argument('--video_path', type=str, default='video_record.avi', help='Path for image to detect')
    parser.add_argument('--model_path', type=str, default='./RetinaFace_Pytorch/model.pt', help='Path for model')
    # parser.add_argument('--save_path', type=str, default='./out/result.avi', help='Path for result image')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--scale', type=float, default=1.0, help='Image resize scale', )
    parser.add_argument('--face_limit', default=None, help='Limit of face detect in a image')
    
    args = parser.parse_args()
    
    conf = get_config(False)
    # args.score = True

    start_time = time.time()

    # Create torchvision model
    return_layers = {'layer2': 1,'layer3': 2,'layer4': 3}
    RetinaFace = torchvision_model.create_retinaface(return_layers)

    # Load trained model
    retina_dict = RetinaFace.state_dict()
    pre_state_dict = torch.load(args.model_path)
    pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
    RetinaFace.load_state_dict(pretrained_dict)

    RetinaFace = RetinaFace.cuda()
    RetinaFace.eval()
    print('Retinaface loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    # learner.threshold = 1.2
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
    if args.update:
        # targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        targets, names = prepare_facebank(conf, learner.model, RetinaFace, tta=args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')
    print(names)

    bboxes, faces = eval_widerface.align_multi(input_img, RetinaFace, conf.face_limit, 16)

    # cap = cv2.VideoCapture(str(conf.facebank_path/args.file_name))
    
    # cap.set(cv2.CAP_PROP_POS_MSEC, args.begin * 1000)
    
    # fps = 15
    # width = 1920
    # height = 1080
    # save_unit = 25
    # video_writer = cv2.VideoWriter('./data/detect_videos/' + str(args.save_name) + '.avi',
    #                                cv2.VideoWriter_fourcc(*'MJPG'), int(fps), (width, height))
    # video_writer_list = []
    # for i, name in enumerate(names):
    #     video_writer_list[i] = cv2.VideoWriter('./data/detect_videos/%s/%s_%s.avi' % name %name %i, cv2.VideoWriter_fourcc(*'MJPG'), int(fps), (width, height))
    # thuan = cv2.VideoWriter('./data/detect_videos/thuan/thuan095.avi', cv2.VideoWriter_fourcc(*'MJPG'), int(fps), (width, height))

    # if args.duration != 0:
    #     i = 0

    # frame_count = 0    
    # while cap.isOpened():
    #     isSuccess, frame = cap.read()
    #     if not isSuccess:
    #         break
    #     frame_count += 1
    #     if frame_count % save_unit != 0:
    #         # print("Not written frame: ", frame_count)
    #         continue
    #     frame = cv2.resize(frame, (width, height))
    #     img = Image.fromarray(frame)
    #     img = np.asarray(img)
    #     img = torch.from_numpy(img)
    #     img = img.permute(2, 0, 1)

    #     input_img = img.unsqueeze(0).float().cuda()
    #     bboxes, faces = eval_widerface.align_multi(input_img, RetinaFace, conf.face_limit, 16)
    #     # for face in faces:
    #     #     face = face.save('face.jpg')

    #     np_img = img.cpu().permute(1,2,0).numpy()
    #     np_img.astype(int)
    #     img = np_img.astype(np.uint8)
    #     frame = img

    #     if bboxes is not None:
    #         results, score = learner.infer(conf, faces, targets, True)
    #         for idx, bbox in enumerate(bboxes):
    #             if names[results[idx] + 1] == 'Wanted':
    #                 # thuan.write(frame)
    #                 color = (107, 235, 255)
    #             else:
    #                 color = (0, 255, 0)
    #             if args.score:
    #                 frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame, color)
    #             else:
    #                 frame = draw_box_name(bbox, names[results[idx] + 1], frame, color)
    #     video_writer.write(frame)
    #     print("written frame: ", frame_count)
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # print("FPS: ", frame_count/(time.time() - start_time))
    # cap.release()
    # video_writer.release()
    # # thuan.release()
    # cv2.destroyAllWindows()

