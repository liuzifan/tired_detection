from imutils.video import VideoStream
import numpy as np
import imutils
from cv2 import cv2
import sys
from PIL import Image

# detection function
def tire_detection(vs):
    closed_frame_num = 0
    # set threshhold of frame 
    # eg: if 5 continuous frames are closed eyes, it is set tired
    thresh = 5
    is_tired = False

    while True:
        frame = vs.read()
        if frame is not None:
            frame = imutils.resize(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = detector.detect_faces(frame) 
            if len(results) == 0:
                # print("no face")
                continue
            elif len(results) > 1:
                print('more than one face')
                continue
            # if there is only one face, then do checking
            else:
                # cut face
                item = results[0]
                box = item['box']
                conf = item['confidence']
                if conf > 0.95 and box[3] > 50 and box[2] > 50 and box[0] >= 0 and box[1] >= 0:
                    face_rgb = frame_rgb[box[1]: box[1] + box[3], box[0]: box[0] + box[2], :] 

                    with torch.no_grad():
                        # use model to detect
                        net.eval()
                        image = Image.fromarray(face_rgb).convert('RGB') 
                        image = transform(image)
                        image = image.reshape(1, *image.shape)
                        out = net(image)
                        pre_label_onehot = torch.softmax(out, 1)
                        topk_values, topk_indices = pre_label_onehot.topk(1, dim=1)
                        topk_indices = topk_indices[0]
                        topk_values = topk_values[0]

                        class_id = topk_indices[0]
                        p = topk_values[0]
                        class_name = class_list[class_id%2]

                        
                        # use predicted result for detecting
                        if class_name == 'close':
                            closed_frame_num += 1
                            if closed_frame_num >= thresh:
                                is_tired = True
                        elif class_name == 'open':
                            closed_frame_num = 0

                        cv2.putText(frame,  class_name + ' : ' + str(p.item()),  (100,  90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),  2)

            if is_tired:
                cv2.putText(frame,  'Tired',  (100,  150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),  2)
                is_tired = False
            else:
                cv2.putText(frame,  'Not Tired',  (100,  150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),  2)
            
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.imshow("Frame", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
    cv2.destroyAllWindows()
    vs.stop()

# main
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# load pre-trained model for detecting
model_path = './epochs/epoch_20.pth'
net = models.resnet18(pretrained=False)
net.avgpool =nn.AdaptiveAvgPool2d((1, 1))
net.fc = nn.Linear(in_features=512, out_features=2, bias=True)
net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor() ])
class_list = ['close', 'open']

# do detecting
vs = VideoStream(src=0).start()
tire_detection(vs)
