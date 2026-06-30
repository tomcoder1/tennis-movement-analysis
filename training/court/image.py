import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import BallTrackerNet
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps

def test_single_image(model_path, img_path, use_refine=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BallTrackerNet(out_channels=15).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Không tìm thấy ảnh tại: {img_path}")
        return
    
    orig_h, orig_w = image.shape[:2]
    # Kích thước chuẩn cho model
    MODEL_W, MODEL_H = 640, 360
    scale_x = orig_w / MODEL_W
    scale_y = orig_h / MODEL_H

    # Tiền xử lý: Resize -> BGR sang RGB -> /255
    img_res = cv2.resize(image, (MODEL_W, MODEL_H))
    img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
    inp = (img_rgb.astype(np.float32) / 255.)
    inp = torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp.float())[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

    points = []
    for kps_num in range(14):
        heatmap = (pred[kps_num] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, scale=1, low_thresh=170, max_radius=25)
        
        if x_pred is not None and y_pred is not None:
            x_final = int(x_pred * scale_x)
            y_final = int(y_pred * scale_y)
            if use_refine and kps_num not in [8, 12, 9]:
                x_final, y_final = refine_kps(image, int(y_final), int(x_final))
            cv2.circle(image, (x_final, y_final), radius=8, color=(0, 0, 255), thickness=-1)

    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Thay đường dẫn model và ảnh của bạn ở đây
test_single_image('/content/drive/MyDrive/Tennis_Project/weights/model_epoch_32.pth', '/content/test_img.jpg')