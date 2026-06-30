import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from model import BallTrackerNet
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps

def process_video():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--use_refine_kps', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BallTrackerNet(out_channels=15).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(args.input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Dùng file tạm để tránh lỗi codec
    temp_raw = 'temp_raw.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(temp_raw, fourcc, fps, (orig_w, orig_h))

    MODEL_W, MODEL_H = 640, 360
    scale_x = orig_w / MODEL_W
    scale_y = orig_h / MODEL_H

    print(f"🎬 Đang xử lý video ({device})...")

    with torch.no_grad():
        for _ in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret: break
            
            # Logic xử lý y hệt code ảnh tĩnh
            img_res = cv2.resize(frame, (MODEL_W, MODEL_H))
            img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
            inp = (img_rgb.astype(np.float32) / 255.)
            inp = torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0).to(device)

            out = model(inp.float())[0]
            pred = F.sigmoid(out).detach().cpu().numpy()

            for i in range(14):
                heatmap = (pred[i] * 255).astype(np.uint8)
                x_p, y_p = postprocess(heatmap, scale=1, low_thresh=170, max_radius=25)
                
                if x_p is not None and y_p is not None:
                    xf = int(x_p * scale_x)
                    yf = int(y_p * scale_y)
                    if args.use_refine_kps and i not in [8, 12, 9]:
                        xf, yf = refine_kps(frame, yf, xf)
                    cv2.circle(frame, (xf, yf), radius=6, color=(0, 0, 255), thickness=-1)

            out_video.write(frame)

    cap.release()
    out_video.release()

    # Nén lại bằng FFmpeg để xem được trên web/máy tính
    os.system(f"ffmpeg -y -i {temp_raw} -vcodec libx264 -crf 23 {args.output_path} > /dev/null 2>&1")
    if os.path.exists(temp_raw): os.remove(temp_raw)
    print(f"🎉 Xong! Kết quả: {args.output_path}")

if __name__ == '__main__':
    process_video()