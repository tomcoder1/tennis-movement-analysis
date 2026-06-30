# TennisCourtDetector

Deep learning network for detecting tennis court.

It was developed a deep learning network to detect tennis court keypoints from broadcast videos. The proposed heatmap-based deep learning network allows to detect 14 points of tennis court. Postprocessing techniques (based on classical computer vision methods) were implemented to enhance net predictions.

![](imgs/dataset_example.png)

---

# Dataset

The dataset consists of 8841 images, which were separated into a train set (75%) and validation set (25%). Each image has 14 annotated points. The resolution of images is 1280×720. This dataset contains all court types (hard, clay, grass).

Dataset download:
https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu/view?usp=drive_link

## Dataset collection

This dataset was created in a semi-automated way. Video highlights from different tournaments with lengths from 2 to 3 minutes were downloaded from YouTube. Frames from videos were extracted every 50 frames and processed using classical computer vision algorithms. The resulting images were manually filtered.

---

# Model architecture

The proposed deep learning network is very similar to TrackNet architecture.

![](imgs/tracknet_arch.png)

The difference is that the input tensor consists of one image instead of three, and the output tensor has 15 channels (14 keypoints + 1 additional center point of the tennis court). The additional point improves convergence during training.

Input and output resolution: 640×360.

---

# Postprocessing

Two postprocessing techniques were implemented.

## 1. Refine keypoints using classical computer vision

![](imgs/net_prediction.png)

Predicted keypoints may not perfectly align with court lines.

![](imgs/crop_example.png)

To improve this, white pixels are extracted from crops, court lines are detected, and intersections are computed using classical computer vision methods.

![](imgs/kps_refine.png)

---

## 2. Use homography to reconstruct shifted keypoints

![](imgs/homography.png)

Predicted points are compared with reference court points using homography transformation. This helps reconstruct missing or shifted points caused by occlusion.

![](imgs/homography_example.png)

---

# Evaluation (metrics)

A keypoint is considered correctly detected if the Euclidean distance between prediction and ground truth is below 7 pixels.

| Method                             | Precision | Accuracy | Median dist |
|-----------------------------------|-----------|-----------|-------------|
| Base model (BM)                  | 0.936     | 0.933     | 2.83        |
| BM + refining kps                | 0.939     | 0.936     | 2.23        |
| BM + homography                  | 0.961     | 0.959     | 2.27        |
| BM + refining kps + homography   | 0.963     | 0.961     | 1.83        |

---

# Pretrained model

Weights download:
https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view?usp=drive_link

---

# How to train

1. Clone repository

```bash
git clone https://github.com/yastrebksv/TennisCourtDetector.git