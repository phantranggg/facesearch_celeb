import utils
import numpy as np
import torch
import torch.nn as nn
import os
from PIL import Image
from tqdm import tqdm
import torchvision.ops as ops
from RetinaFace_Pytorch.align_trans import get_reference_facial_points, warp_and_crop_face

def get_detections(img_batch, model, score_threshold=0.5, iou_threshold=0.5):
    # model.eval()
    with torch.no_grad():
        classifications, bboxes, landmarks = model(img_batch)
        batch_size = classifications.shape[0]
        picked_boxes = []
        picked_landmarks = []
        picked_scores = []
        
        for i in range(batch_size):
            classification = torch.exp(classifications[i,:,:])
            bbox = bboxes[i,:,:]
            landmark = landmarks[i,:,:]

            # choose positive and scores > score_threshold
            scores, argmax = torch.max(classification, dim=1)
            argmax_indice = argmax==0
            scores_indice = scores > score_threshold
            positive_indices = argmax_indice & scores_indice
            
            scores = scores[positive_indices]

            if scores.shape[0] == 0:
                picked_boxes.append(None)
                picked_landmarks.append(None)
                picked_scores.append(None)
                continue

            bbox = bbox[positive_indices]
            landmark = landmark[positive_indices]

            keep = ops.boxes.nms(bbox, scores, iou_threshold)
            keep_boxes = bbox[keep]
            keep_landmarks = landmark[keep]
            keep_scores = scores[keep]
            keep_scores.unsqueeze_(1)
            picked_boxes.append(keep_boxes)
            picked_landmarks.append(keep_landmarks)
            picked_scores.append(keep_scores)
        
        return picked_boxes, picked_landmarks, picked_scores

def align(img, RetinaFace):
    boxes, landmarks, scores = get_detections(img, RetinaFace, score_threshold=0.5, iou_threshold=0.3)
    # _, landmarks = self.detect_faces(img)
    facial5points = [[landmarks[0][j],landmarks[0][j+5]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, get_reference_facial_points(default_square= True), crop_size=(112,112))
    return Image.fromarray(warped_face)

def align_multi(conf, img, RetinaFace, limit=None, min_face_size=30.0):
    img = torch.from_numpy(np.asarray(img))
    img = img.permute(2, 0, 1)            # if img is GRB then change to BGR
    img = img.unsqueeze(0).float().to(conf.device)
    boxes, landmarks, scores = get_detections(img, RetinaFace, score_threshold=0.5, iou_threshold=0.3)

    # print(len(boxes))
    # if limit:
        # boxes = boxes[:limit]
        # landmarks = landmarks[:limit]
        # print(len(landmarks[0]))
    # print(boxes)
    img = img.squeeze(0).to(conf.device)
    img = img.cpu().permute(1,2,0).numpy()
    faces = []
    for j, boxes in enumerate(boxes):
        if boxes is not None:
            for box, landmark, score in zip(boxes, landmarks[j], scores[j]):
                facial5points = [[landmark[j*2], landmark[j*2+1]] for j in range(5)]
                warped_face = warp_and_crop_face(img, facial5points, get_reference_facial_points(default_square= True), crop_size=(112,112))
                # warped_face.astype(int)
                warped_face = warped_face.astype(np.uint8)
                faces.append(Image.fromarray(warped_face))
    return boxes, faces

def compute_overlap(a,b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    # (N, K) ndarray of overlap between boxes and query_boxes
    return torch.from_numpy(intersection / ua)    


def evaluate(val_data,retinaFace,threshold=0.5):
    recall = 0.
    precision = 0.
    #for i, data in tqdm(enumerate(val_data)):
    for data in tqdm(iter(val_data)):
        img_batch = data['img'].cuda()
        annots = data['annot'].cuda()


        picked_boxes,_,_ = get_detections(img_batch,retinaFace)
        recall_iter = 0.
        precision_iter = 0.

        for j, boxes in enumerate(picked_boxes):          
            annot_boxes = annots[j]
            annot_boxes = annot_boxes[annot_boxes[:,0]!=-1]

            if boxes is None and annot_boxes.shape[0] == 0:
                continue
            elif boxes is None and annot_boxes.shape[0] != 0:
                recall_iter += 0.
                precision_iter += 1.
                continue
            elif boxes is not None and annot_boxes.shape[0] == 0:
                recall_iter += 1.
                precision_iter += 0.   
                continue         
            
            overlap = ops.boxes.box_iou(annot_boxes, boxes)
                 
            # compute recall
            max_overlap, _ = torch.max(overlap,dim=1)
            mask = max_overlap > threshold
            detected_num = mask.sum().item()
            recall_iter += detected_num/annot_boxes.shape[0]

            # compute precision
            max_overlap, _ = torch.max(overlap,dim=0)
            mask = max_overlap > threshold
            true_positives = mask.sum().item()
            precision_iter += true_positives/boxes.shape[0]

        recall += recall_iter/len(picked_boxes)
        precision += precision_iter/len(picked_boxes)

    return recall/len(val_data),precision/len(val_data)
