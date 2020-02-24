import torch
import mobilenet_v1
import torchvision.transforms as transforms
from utils.ddfa import ToTensorGjz, NormalizeGjz
from utils.inference import crop_img, predict_68pts

def load_pretrain_model():
    checkpoint_fp = 'phase1_wpdc_vdc.pth.tar'
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

def detect_pose(img_ori, target, model):
    STD_SIZE = 120
    pose_list = []
    for rect in target:
        roi_box = rect[0:4]
        # print(roi_box)
        try:
          img = crop_img(img_ori, roi_box)
        except Exception as e:
          print(e)
          continue

        shape_img = img.shape
        if shape_img[0] < 100 or shape_img[1] < 100:
          continue
        if len(img) == 0:
          return pose_list
        try:
          img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
          print(e)
          continue
        transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
        input = transform(img).unsqueeze(0)
        with torch.no_grad():
            param = model(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        pts68 = predict_68pts(param, roi_box)
        P, pose, pose_angle = parse_pose(param)
          
        # if shape_img[0]>100 and shape_img[1]>100 and np.abs(pose_angle[0]) <= 60 and np.abs(pose_angle[1]) <= 60 and np.abs(pose_angle[2])<= 60:
        #   print(shape_img)
        #   plt.imshow(img)
        #   plt.show()
        pose_list.append(pose_angle)
    return pose_list


if __name__ == '__main__':
    model_3ddfa = load_pretrain_model()

    for i in range(1):
      img, target = get_bbox(i, imgs_path, words)
      # img_raw = visualization(img, target)
      pose_list = detect_pose(img, target, model_3ddfa)
      # cv2.imwrite('test.png', img_raw)
      print(pose_list)