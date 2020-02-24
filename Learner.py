from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
# from tensorboardX import SummaryWriter
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math

class face_learner(object):
    def __init__(self, conf, inference=False):
        # print(conf)
        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        
        if not inference:
            self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)        

            # self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
            
            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            else:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            print(self.optimizer)
#             self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

            print('optimizers generated')    
            self.board_loss_every = len(self.loader)//100
            self.evaluate_every = len(self.loader)//10
            self.save_every = len(self.loader)//5
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(self.loader.dataset.root.parent)
        else:
            self.threshold = conf.threshold
    
    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        torch.save(
            self.model.state_dict(), save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
    
    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path            
        # self.model = torch.nn.DataParallel(self.model)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str)))
        else:
            self.model.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str), map_location='cpu'))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path/'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path/'optimizer_{}'.format(fixed_str)))
        
    # def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
    #     self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
    #     self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
    #     self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
#         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
#         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
#         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)
        
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
                # embs.append((emb + emb_mirror)/2)
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum          

    def infer_one(self, conf, face, target_embs, tta=False):
        embs = []
        if tta:
            mirror = trans.functional.hflip(face)
            emb = self.model(conf.test_transform(face).to(conf.device).unsqueeze(0))
            emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
            embs.append(l2_norm(emb + emb_mirror))
        else:
            embs.append(self.model(conf.test_transform(face).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)

        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum 

    def embedding(self, conf, img, tta=False):
        if tta:
            mirror = trans.functional.hflip(img)
            emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
            emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
            return l2_norm(emb + emb_mirror)
        else:                        
            return self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
