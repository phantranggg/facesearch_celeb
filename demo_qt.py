from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import faiss, time, os, sys
import pickle5 as pickle
import numpy as np
import cv2
import torch
import dill
# import math
from PIL import Image
from config import get_config
from Learner import face_learner
from RetinaFace_Pytorch import torchvision_model, eval_widerface

conf = get_config(False)

start = time.time()
# Create torchvision model
return_layers = {'layer2': 1,'layer3': 2,'layer4': 3}
RetinaFace = torchvision_model.create_retinaface(return_layers)

# Load trained model
retina_dict = RetinaFace.state_dict()
if torch.cuda.is_available():
    pre_state_dict = torch.load('./RetinaFace_Pytorch/model.pt')
else:
    pre_state_dict = torch.load('./RetinaFace_Pytorch/model.pt', map_location='cpu')
pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
RetinaFace.load_state_dict(pretrained_dict)

RetinaFace = RetinaFace.to(conf.device)
RetinaFace.eval()
print('Retinaface loaded')

learner = face_learner(conf, True)
learner.load_state(conf, 'final.pth', True, True)
learner.model.eval()
print('learner loaded')

print("Load pretrain: ", time.time() - start)

start = time.time()
# torch.cuda.empty_cache()
with open('filename_list.pkl', 'rb') as f:
    filename_list = pickle.load(f)
print(filename_list[0:10])
print("Load filename pickle: ", time.time() - start)

form_class = uic.loadUiType("ui_example/search.ui")[0]

dimention = 512
neareast_neighbor = 4

# build index
start = time.time()
index = faiss.IndexFlatL2(dimention)
with open('emb_list.pkl', 'rb') as f:
    emb_list = pickle.load(f)
index.add(emb_list)
print(index.is_trained, index.ntotal)
print('Load emb list + indexing: ', time.time() - start)
            
class DisplayImage(QWidget):
    def __init__(self, parent=None):
        super(DisplayImage, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()


class MyWindowClass(QMainWindow, form_class):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.selectBtn.clicked.connect(self.select_clicked)

        self.resultLb.setText('')
        self.resultLb1.setText('')
        for scoreLb in [self.scoreLb, self.scoreLb2, self.scoreLb3, self.scoreLb4]:
            scoreLb.setText('')

        self.big_window_width = self.widget1.frameSize().width()
        self.big_window_height = self.widget1.frameSize().height()
        self.widget1 = DisplayImage(self.widget1)
        self.widget2 = DisplayImage(self.widget2)

        img = cv2.imread(os.path.expanduser('./data/default.png'))
        self.display_img(img, self.big_window_width, self.big_window_height, self.widget1)
        self.display_img(img, self.big_window_width, self.big_window_height, self.widget2)
        
        self.small_window_width = self.widget3.frameSize().width()
        self.small_window_height = self.widget3.frameSize().height()
        self.widget3 = DisplayImage(self.widget3)
        self.widget4 = DisplayImage(self.widget4)
        self.widget5 = DisplayImage(self.widget5)

        self.display_img(img, self.small_window_width, self.small_window_height, self.widget3)
        self.display_img(img, self.small_window_width, self.small_window_height, self.widget4)
        self.display_img(img, self.small_window_width, self.small_window_height, self.widget5)

    def display_img(self, img, window_width, window_height, obj):
        img_height, img_width, img_colors = img.shape
        scale_w = float(window_width) / float(img_width)
        scale_h = float(window_height) / float(img_height)
        scale = min([scale_w, scale_h])
        if scale == 0:
            scale = 1
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, bpc = img.shape
        bpl = bpc * width
        image = QImage(img.data, width, height, bpl, QImage.Format_BGR888)
        obj.setImage(image)

    def select_clicked(self):
        image_path, _filter = QFileDialog.getOpenFileName(None, 'Open File', '', 'Image File (*.jpg *.jpeg)')
        image_path = str(image_path)
        img = cv2.imread(os.path.expanduser(image_path))
        if img is None:
            return
        self.display_img(img, self.big_window_width, self.big_window_height, self.widget1)
        self.search(image_path)
    
    # def score_cal(self, dist, threshold):
    #     if (dist < threshold/2):
    #         return 1 - 0.2/threshold*dist
    #     elif (dist < threshold):
    #         return ((dist-threshold)*(dist-threshold/2)*2 - 0.9*4*dist*(dist-threshold) + dist*(dist-threshold /2))/(threshold**2)
    #     else:
    #         return 1/(1+math.exp(3*(dist-threshold)))

    def score_cal(self, dist, threshold):
        if dist <= threshold:
            if dist/threshold >= 0.5:
                res = 70 + (1 - dist/threshold) * 60
            else:
                res = 90 + (1 - dist/threshold) * 10
        elif dist > threshold:
            res = 60 - (dist/threshold - 1) * 60
        return res

    def display_score(self, dists):
        threshold = 1.1
        for i, scoreLb in enumerate([self.scoreLb, self.scoreLb2, self.scoreLb3, self.scoreLb4]):
            score = self.score_cal(dists[i], threshold)
            scoreLb.setText('{:.2f}%'.format(score))
            # scoreLb.setText(int(score*100))
        self.resultLb.setText('FOUND IT!')
        self.resultLb1.setText('Similarity Score: ')

    def search(self, image_path):
        # crop face from input image, then embed
        start = time.time()
        img = Image.open(image_path)
        w, h = img.size
        if w >= 1000 or h >= 1000:
            img = img.resize((int(w/2), int(h/2)), Image.NEAREST)
        elif w >= 2000 or h >= 2000:
            img = img.resize((int(w/3), int(h/3)), Image.NEAREST)
        bboxes, faces = eval_widerface.align_multi(conf, img, RetinaFace, 1, 16)
        if len(faces) == 0:
            print("Can't find face in input image")
            return
        query = learner.embedding(conf, faces[0], True)
        query = query.cpu().detach().numpy()

        # display all neareast distances & indexes
        D, I = index.search(query, neareast_neighbor)
        print(I)
        print(D)   
        print('Find time: ', time.time() - start) 

        threshold = 1.105
        not_display_threshold = 1.2
        dists = D[0]
        
        # display widget + score for nearest neighbor
        self.resultLb.setText('FOUND IT!')
        self.resultLb1.setText('Similarity Score: ')
        score = self.score_cal(dists[0], threshold)
        self.scoreLb.setText('{:.2f}%'.format(score))

        print(os.path.expanduser(filename_list[I[0, 0]]))
        img = cv2.imread(os.path.expanduser(filename_list[I[0, 0]]))
        self.display_img(img, self.big_window_width, self.big_window_height, self.widget2)

        # display 3 small widgets + score
        scores = []
        for i, scoreLb in enumerate([self.scoreLb2, self.scoreLb3, self.scoreLb4]):
            if dists[i+1] > not_display_threshold:
                score = None
                scoreLb.setText("")
            else:
                score = self.score_cal(dists[i+1], threshold)
                scoreLb.setText('{:.2f}%'.format(score))
            scores.append(score)
        for i, obj in enumerate([self.widget3, self.widget4, self.widget5]):
            if scores[i] != None:
                img = cv2.imread(os.path.expanduser(filename_list[I[0, i+1]]))
            else:
                img = cv2.imread(os.path.expanduser('./data/default.png'))
            self.display_img(img, self.small_window_width, self.small_window_height, obj)

    # def closeEvent(self, event):
    #     global running
    #     running = False


app = QApplication(sys.argv)
w = MyWindowClass(None)
w.setWindowTitle('Face Search')
w.show()
app.exec_()

