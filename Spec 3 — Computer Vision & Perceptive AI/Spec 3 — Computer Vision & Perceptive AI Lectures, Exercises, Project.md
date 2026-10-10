# Spec 3 — Computer Vision & Perceptive AI: Lectures, Exercises, Project

Oct 7, 2026 · @Dang Tu

## How to use this pack

This pack covers the 10 handbook topics of Spec 3 as one lecture each, with exercises and a real-world project. Lectures 1–2 are classical geometry and need no deep learning; from Lecture 3 on, keep pace with Core 3 (CNNs are Core 3 Lecture 7). Spec 4 Robotics in semester 2 requires this module.

**Handbook grading:** assignments 20% (vision pipelines on standard image and video datasets), group project 30%, closed-book exam 50%.

**Exercise levels:** (A) by hand, exam style; (B) code with OpenCV and PyTorch; (C) stretch.

| Resource | Use it for |
| --- | --- |
| Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed. (free at szeliski.org/Book) | Main text |
| Hartley & Zisserman, *Multiple View Geometry* | Lectures 1, 2, 7 depth |
| Prince, *Computer Vision: Models, Learning, and Inference* (free PDF) | Probabilistic view |
| Stanford CS231n and Michigan EECS 498 (Justin Johnson) lectures | Deep-learning lectures 3–5 |
| First Principles of Computer Vision (Shree Nayar, YouTube) | Lectures 1, 2, 6, 7 intuition |

**Setup:** `opencv-python`, `numpy`, `torch`, `torchvision`, `timm`, `ultralytics` (YOLO), `open3d` (3D), `matplotlib`. Colab or Kaggle GPU is enough for exercises; use the faculty GPU cluster for the project.

**Datasets you will meet:** ImageNet, COCO, Cityscapes, KITTI, MOT17, plus Vietnamese traffic datasets for the project.

## Lecture 1 — Image formation, camera models and projective geometry

**Goal:** model how a 3D point becomes a pixel, correct lens distortion, and calibrate a real camera. Reading: Szeliski ch. 2.1–2.3, 11.1; Hartley & Zisserman ch. 2, 6; Zhang, "A Flexible New Technique for Camera Calibration" (2000).

### 1.1 Homogeneous coordinates

A 2D point (x, y) is written (x, y, 1), and (kx, ky, k) is the same point for any k ≠ 0. This lets translation, rotation and perspective projection all be matrix multiplications, and represents points at infinity (vanishing points) as (x, y, 0). A line l and a point p meet when lᵀp = 0; the line through two points is their cross product.

### 1.2 The pinhole camera

```latex
\lambda \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = K \,[\,R \mid t\,] \begin{pmatrix} X \\ Y \\ Z \\ 1 \end{pmatrix}, \qquad K = \begin{pmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}
```

- **Extrinsics** R, t: where the camera is and which way it points (world → camera).
- **Intrinsics** K: focal lengths in pixels (fₓ, fᵧ), principal point (cₓ, cᵧ), skew s (usually 0).
- P = K\[R | t\] is the 3×4 projection matrix with 11 degrees of freedom.

Perspective effects: far objects look smaller (division by depth Z), parallel lines meet at vanishing points, and depth is lost: every point on a ray maps to the same pixel.

### 1.3 Lens distortion

Real lenses bend straight lines, especially wide-angle and traffic cameras. Radial distortion: x\_d = x(1 + k₁r² + k₂r⁴ + k₃r⁶); tangential terms p₁, p₂ handle a tilted lens. Undistort images before measuring anything geometric.

### 1.4 Homographies

A planar surface (road, document, football pitch) maps between two images by a 3×3 homography H (8 degrees of freedom), estimated from ≥ 4 point pairs with the direct linear transform (DLT, an SVD problem, Core 1 L1). Uses: bird's-eye view of a road for measuring vehicle speed, document scanning, panorama stitching.

### 1.5 Calibration

Zhang's method: photograph a checkerboard of known square size in 10–20 poses; detect corners; estimate a homography per view; solve for K in closed form; then refine K, distortion and all poses by minimising reprojection error with Levenberg–Marquardt. A good calibration has a mean reprojection error below \~0.5 pixel. OpenCV: `findChessboardCorners`, `calibrateCamera`, `undistort`.

### 1.6 Light and colour (brief)

Pixel values depend on illumination, surface reflectance (Lambertian vs specular) and sensor response. Colour spaces: RGB, HSV (separates hue from brightness), Lab (perceptually uniform). Gamma correction and white balance affect any model trained on images, which is why illumination changes hurt real-world performance.

### Exercises

1. (A) Project the 3D point (1, 2, 10) with fₓ = fᵧ = 800, c = (320, 240), R = I, t = 0.
2. (A) Find the vanishing point of the 3D line direction (1, 0, 1) for the camera above.
3. (A) Count the degrees of freedom of P and of H, and explain the minimum number of correspondences for each.
4. (A) Derive the DLT equations for a homography from one point correspondence.
5. (B) Calibrate your phone camera with a printed checkerboard; report K, distortion coefficients and reprojection error; undistort a wide-angle photo.
6. (B) Pick 4 points on a road in a street photo and compute a bird's-eye view with `getPerspectiveTransform` and `warpPerspective`.
7. (C) Implement DLT for homography estimation with point normalisation and compare against OpenCV.

## Lecture 2 — Feature detection, description, matching and robust estimation

**Goal:** find the same physical points in different images and estimate geometry despite many wrong matches. Reading: Szeliski ch. 3.2, 7.1–7.2, 8.1; Lowe, "Distinctive Image Features from Scale-Invariant Keypoints" (2004); Fischler & Bolles, "RANSAC" (1981).

### 2.1 Filtering and gradients

Convolution with small kernels: Gaussian blur (removes noise), Sobel (image gradients Iₓ, Iᵧ), Laplacian of Gaussian (blobs). The gradient magnitude and direction are the raw material for edges and features. Canny edge detection = Gaussian smoothing + gradient + non-maximum suppression + hysteresis thresholds.

### 2.2 Corner detection

A corner changes appearance when shifted in any direction. The structure tensor over a window W:

```latex
M = \sum_{(x,y) \in W} \begin{pmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{pmatrix}, \qquad R = \det M - k\, (\mathrm{tr}\, M)^2
```

Two large eigenvalues of M mean a corner, one large means an edge, both small mean flat. Harris uses R (k ≈ 0.04–0.06); Shi–Tomasi uses the smaller eigenvalue directly.

### 2.3 Scale and rotation invariance

| Detector / descriptor | Invariance | Notes |
| --- | --- | --- |
| SIFT | Scale, rotation, some illumination | DoG scale space; 128-D histogram of gradient orientations |
| SURF | Scale, rotation | Faster SIFT approximation |
| ORB | Rotation, scale via pyramid | FAST corners + binary BRIEF descriptor; free and real-time; used in ORB-SLAM |
| SuperPoint + SuperGlue / LightGlue | Learned | State of the art under large viewpoint and lighting change |

### 2.4 Matching

Compare descriptors with Euclidean distance (SIFT) or Hamming distance (ORB). Lowe's ratio test keeps a match only if the best distance is < 0.7–0.8 × the second best, rejecting ambiguous matches. Mutual nearest-neighbour checks help further. Use FLANN or FAISS for large sets.

### 2.5 RANSAC

Even after filtering, many matches are wrong. RANSAC: repeatedly sample the minimal set (4 pairs for a homography, 8 or 5 for the fundamental/essential matrix), fit a model, count inliers within a threshold, keep the best, then refit on all its inliers. Iterations needed for success probability p with inlier ratio w and sample size s:

```latex
N = \frac{\log(1 - p)}{\log(1 - w^s)}
```

For p = 0.99, w = 0.5, s = 4: N ≈ 72. Variants: MSAC, PROSAC, MAGSAC++ (OpenCV's `USAC_MAGSAC`).

### Exercises

1. (A) Compute M and the Harris response for a given 3×3 patch of gradients; classify it as corner, edge or flat.
2. (A) Explain why SIFT is scale-invariant but Harris is not.
3. (A) Compute N for RANSAC with w = 0.3, s = 8, p = 0.99. What does that imply for estimating the fundamental matrix with many outliers?
4. (A) Show why the ratio test removes matches in repetitive textures (windows of a building).
5. (B) Implement Harris from scratch and compare with `cv2.cornerHarris` on a photo of the VGU campus.
6. (B) Stitch a 3-photo panorama: ORB or SIFT features, ratio test, RANSAC homography, warping and blending.
7. (B) Measure matching accuracy of SIFT, ORB and SuperPoint+LightGlue on the HPatches benchmark under viewpoint and illumination changes.
8. (C) Implement RANSAC for a homography yourself and plot inliers vs iterations.

## Lecture 3 — Deep learning for vision: CNN and vision-transformer backbones

**Goal:** choose and adapt a backbone network for a vision task, knowing its accuracy, cost and inductive biases. Reading: Szeliski ch. 5.3–5.5; Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT, 2021); Liu et al., "Swin Transformer" (2021); Liu et al., "A ConvNet for the 2020s" (2022). Core 3 Lecture 7 covers CNN basics.

### 3.1 What a backbone does

A backbone turns an image into feature maps at several resolutions (strides 4, 8, 16, 32). Task heads (classification, detection, segmentation) sit on top. Early layers capture edges and textures, later layers object parts and categories. Feature pyramid networks (FPN) merge levels so small and large objects both get strong features.

### 3.2 CNN backbones (recap and extension)

ResNet (residual blocks), EfficientNet (balanced scaling of depth, width and resolution), MobileNetV3 (depthwise separable convs for phones), ConvNeXt (modern CNN with large kernels, LayerNorm and GELU). CNNs have built-in locality and translation equivariance, so they learn well from moderate data.

### 3.3 Vision transformers

ViT splits the image into 16×16 patches, embeds each linearly, adds position embeddings and a \[CLS\] token, and runs a transformer encoder (Core 3 L8).

```latex
N_{\text{tokens}} = \frac{H \cdot W}{P^2}, \qquad \text{e.g. } \frac{224 \cdot 224}{16^2} = 196
```

ViTs have weak inductive biases, so they need large pre-training data (ImageNet-21k, JFT, or self-supervised pre-training) but then scale better than CNNs. Global attention costs O(N²), which is expensive at high resolution.

| Backbone | Idea | Strength |
| --- | --- | --- |
| ViT | Plain global attention over patches | Scales with data; simple |
| DeiT | ViT + distillation + strong augmentation | Trains on ImageNet-1k only |
| Swin | Attention within shifted local windows; hierarchical | Dense prediction at high resolution |
| ConvNeXt | CNN redesigned with transformer tricks | Strong, efficient, easy to deploy |
| DINOv2 / CLIP ViT | Self-supervised / image–text pre-trained | Excellent frozen features; zero-shot (CLIP) |

### 3.4 Choosing a backbone

Decide by: accuracy needed, latency and memory on the target device (phone, edge box such as Jetson, server GPU), amount of labelled data, and input resolution. Check FLOPs, parameter count and measured throughput, not just ImageNet top-1. For small datasets, start from strong pretrained frozen features (DINOv2, CLIP) with a linear probe, then fine-tune.

### 3.5 Training tricks for vision

Strong augmentation (RandAugment, Mixup, CutMix), label smoothing, stochastic depth, EMA of weights, AdamW with cosine schedule and warm-up. Most "architecture gains" in papers come partly from these recipes, so compare models under the same recipe.

### Exercises

1. (A) How many tokens does a ViT with 14×14 patches produce for a 518×518 image? What does doubling resolution do to attention cost?
2. (A) Explain why ViTs need more data than CNNs, in terms of inductive bias.
3. (A) Explain how Swin's shifted windows let information cross window borders.
4. (B) Compare ResNet-50, EfficientNet-B0, ConvNeXt-T, ViT-S and a DINOv2 linear probe on Oxford-IIIT Pets: accuracy, parameters, images/s on your GPU.
5. (B) Visualise ViT attention maps (attention rollout) and CNN Grad-CAM for the same images.
6. (B) Measure the latency of MobileNetV3 vs ResNet-50 on a CPU and (if possible) a phone using ONNX Runtime.
7. (C) Use CLIP for zero-shot classification of Vietnamese street-food photos with prompts in English and Vietnamese; compare accuracy.

## Lecture 4 — Object detection: two-stage and single-shot detectors

**Goal:** localise and classify every object in an image, and evaluate detectors correctly. Reading: Szeliski ch. 6.3; Ren et al., "Faster R-CNN" (2015); Redmon et al., "YOLO" (2016); Lin et al., "Focal Loss / RetinaNet" (2017); Carion et al., "DETR" (2020).

### 4.1 The task

Output a set of boxes (x, y, w, h), each with a class and a confidence. Difficulties: unknown number of objects, wide range of sizes, occlusion, crowded scenes (dense motorbike traffic is an extreme case).

### 4.2 Building blocks

- **Intersection over union:** IoU(A, B) = area(A ∩ B) / area(A ∪ B).
- **Anchors:** predefined boxes of several sizes and aspect ratios at each feature-map location; the network predicts class scores and offsets relative to them.
- **Non-maximum suppression (NMS):** sort boxes by score; drop any box with IoU > 0.5–0.7 with a kept box of the same class. Crowds need softer NMS.
- **Box regression losses:** smooth L1 on offsets, or IoU-based losses (GIoU, DIoU, CIoU).

### 4.3 Two-stage detectors (R-CNN family)

| Model | Change |
| --- | --- |
| R-CNN (2014) | Selective-search proposals, a CNN on each crop: slow |
| Fast R-CNN (2015) | One CNN pass; RoI pooling crops features |
| Faster R-CNN (2015) | Region proposal network (RPN) replaces selective search |
| Mask R-CNN (2017) | Adds a mask head, RoIAlign (Lecture 5) |
| Cascade R-CNN | Successive heads with rising IoU thresholds |

Stage 1 proposes \~1,000 candidate regions, stage 2 classifies and refines them. Accurate, especially for small objects, but slower.

### 4.4 Single-shot detectors

YOLO, SSD and RetinaNet predict boxes and classes densely in one pass. The challenge is class imbalance (most locations are background); RetinaNet's focal loss down-weights easy negatives:

```latex
\text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t), \qquad \gamma \approx 2
```

Modern YOLO versions (Ultralytics YOLOv8/YOLO11) are anchor-free, use FPN/PAN necks, decoupled heads and strong augmentation (mosaic), and run in real time on edge devices. DETR-style detectors (DETR, RT-DETR) use a transformer with set prediction and Hungarian matching, which removes NMS.

### 4.5 Evaluation: mAP

For each class, rank detections by confidence; a detection is a true positive if IoU with an unmatched ground-truth box ≥ threshold. Compute the precision–recall curve and its area (AP). mAP averages over classes. COCO mAP also averages over IoU thresholds 0.50:0.05:0.95 and reports AP for small, medium and large objects. Also report speed (FPS or ms per image) on the target hardware.

### Exercises

1. (A) Compute IoU of boxes (0, 0, 10, 10) and (5, 5, 15, 15) (x1, y1, x2, y2).
2. (A) Run NMS by hand on 5 given boxes with scores at IoU threshold 0.5.
3. (A) From 6 ranked detections and 4 ground-truth objects, draw the PR curve and compute AP.
4. (A) Explain why the focal loss helps one-stage detectors and what γ = 0 reduces to.
5. (B) Run a pretrained Faster R-CNN (torchvision) and YOLO11 on 20 Vietnamese street photos; compare boxes, speed and failure cases on motorbikes and pedestrians.
6. (B) Fine-tune YOLO on a small custom dataset you label with Label Studio or CVAT (e.g. 200 images of helmets/no helmets); report mAP@0.5 and mAP@0.5:0.95.
7. (C) Implement NMS and the AP computation yourself and check them against `torchvision.ops.nms` and `pycocotools`.

## Lecture 5 — Semantic, instance and panoptic segmentation

**Goal:** label every pixel, separate individual objects, and evaluate masks. Reading: Szeliski ch. 6.4; Long et al., "Fully Convolutional Networks" (2015); Ronneberger et al., "U-Net" (2015); He et al., "Mask R-CNN" (2017); Kirillov et al., "Segment Anything" (2023).

### 5.1 Three tasks

| Task | Output | Distinguishes instances? | Typical metric |
| --- | --- | --- | --- |
| Semantic | A class per pixel (road, sky, car) | No | mIoU |
| Instance | A mask per countable object | Yes | Mask AP |
| Panoptic | Every pixel: "stuff" class or "thing" instance | Yes, for things | PQ |

"Things" are countable (cars, people); "stuff" is amorphous (road, vegetation, sky).

### 5.2 Semantic segmentation architectures

- **FCN:** replace fully connected layers with convolutions and upsample; skip connections recover detail.
- **U-Net:** symmetric encoder–decoder with skip connections at each level; the standard for medical and satellite images with little data.
- **DeepLab:** atrous (dilated) convolutions keep resolution; atrous spatial pyramid pooling captures multi-scale context.
- **Transformer-based:** SegFormer (efficient), Mask2Former (one architecture for all three tasks).

Loss: per-pixel cross-entropy; add Dice loss for small structures or class imbalance.

### 5.3 Instance segmentation

Mask R-CNN adds a small fully convolutional mask head to Faster R-CNN (Lecture 4) that predicts a 28×28 binary mask per detected box. RoIAlign samples features with bilinear interpolation instead of rounding, which matters for mask quality. One-stage alternatives: YOLACT, SOLO, YOLO-seg.

### 5.4 Foundation models for segmentation

Segment Anything (SAM, SAM 2 for video) produces masks from prompts (points, boxes) with no task-specific training. It does not know class names; combine it with a detector or CLIP for labels. It speeds up annotation enormously: click once, correct the mask.

### 5.5 Metrics

```latex
\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}, \qquad \text{mIoU} = \frac{1}{C}\sum_c \text{IoU}_c, \qquad \text{PQ} = \underbrace{\frac{\sum_{(p,g) \in TP} \text{IoU}(p,g)}{|TP|}}_{\text{segmentation quality}} \times \underbrace{\frac{|TP|}{|TP| + \tfrac12|FP| + \tfrac12|FN|}}_{\text{recognition quality}}
```

Pixel accuracy is misleading when one class (road, background) dominates; report mIoU and per-class IoU. Boundary metrics (boundary F-score) matter where edges matter.

### 5.6 Applications and data masking

Autonomous driving (drivable area, lanes), satellite imagery (flood mapping, land use, rice-field area), medical imaging (tumours, organs), photo editing, and privacy: segment and blur faces, people and licence plates before storing or sharing images (Lecture 10).

### Exercises

1. (A) Compute per-class IoU and mIoU from a given 3-class pixel confusion matrix.
2. (A) Explain why U-Net's skip connections improve boundaries.
3. (A) Compute PQ for a small example with 3 matched segments (IoUs 0.9, 0.8, 0.6), 1 FP and 2 FN.
4. (A) Compute the receptive field gain of a 3×3 convolution with dilation 2 and 4.
5. (B) Train a U-Net on a public satellite or flood dataset (e.g. Sen1Floods11) and report mIoU.
6. (B) Fine-tune SegFormer on a Cityscapes subset; compare with DeepLabv3 for accuracy and speed.
7. (B) Use SAM with box prompts from a detector to segment every motorbike in 20 street photos; measure time per image.
8. (C) Build a privacy filter: detect and segment faces and licence plates in video, then blur them; measure missed detections on 200 frames.

## Lecture 6 — Motion analysis and tracking: optical flow, Kalman filtering, multi-object tracking

**Goal:** estimate how pixels and objects move through video, and keep consistent identities over time. Reading: Szeliski ch. 9; Lucas & Kanade (1981); Teed & Deng, "RAFT" (2020); Welch & Bishop, "An Introduction to the Kalman Filter"; Zhang et al., "ByteTrack" (2022).

### 6.1 Optical flow

Assume brightness stays constant along motion: I(x + u, y + v, t + 1) = I(x, y, t). A first-order Taylor expansion (Core 1 L2) gives the optical-flow constraint:

```latex
I_x u + I_y v + I_t = 0
```

One equation, two unknowns (the aperture problem). **Lucas–Kanade** assumes constant flow in a small window and solves least squares (well-posed at corners, Lecture 2's structure tensor); image pyramids handle large motion. **Horn–Schunck** adds a global smoothness term for dense flow. Deep methods (RAFT) are far more accurate. Uses: video stabilisation, motion segmentation, action recognition, estimating vehicle speed.

### 6.2 Kalman filter

Track a hidden state (e.g. box centre, size and velocities) with a linear motion model and noisy measurements:

- **Predict:** x̂⁻ = F x̂, P⁻ = F P Fᵀ + Q
- **Update:** K = P⁻Hᵀ(H P⁻ Hᵀ + R)⁻¹, x̂ = x̂⁻ + K(z − H x̂⁻), P = (I − KH)P⁻

F is the motion model (constant velocity), H maps state to measurement, Q and R are process and measurement noise covariances. The gain K balances trust in the prediction against the measurement; it is optimal for linear-Gaussian systems (Core 1 L7 Gaussian conditioning). The extended and unscented Kalman filters handle non-linear models; particle filters handle multi-modal beliefs. You will reuse this in SLAM (Lecture 7, Spec 4).

### 6.3 Multi-object tracking (MOT)

Tracking-by-detection: run a detector each frame (Lecture 4), predict each existing track with a Kalman filter, then associate detections to tracks.

| Tracker | Association | Notes |
| --- | --- | --- |
| SORT | IoU cost + Hungarian algorithm (Core 2 L7) | Very fast; identity switches under occlusion |
| DeepSORT | IoU + appearance embedding (re-identification) | Fewer ID switches |
| ByteTrack | Also associates low-confidence detections in a second pass | Strong and simple; good in crowds |
| BoT-SORT / OC-SORT | Camera-motion compensation, observation-centric updates | Better with moving cameras and non-linear motion |

Track management: start a track after k consecutive hits, delete after m misses.

### 6.4 MOT metrics

MOTA = 1 − (FN + FP + ID switches) / ground-truth objects (dominated by detection quality). IDF1 measures how long identities are kept correct. HOTA balances detection and association. Benchmarks: MOT17, MOT20 (crowds), UA-DETRAC (vehicles).

### 6.5 Counting and speed

Count objects crossing a virtual line by checking when a track's centre changes side. Estimate speed by mapping track positions to road coordinates with a homography (Lecture 1) and dividing distance by time.

### Exercises

1. (A) Derive the optical-flow constraint from brightness constancy, and explain the aperture problem with a diagram.
2. (A) Solve Lucas–Kanade for a window with given Iₓ, Iᵧ, Iₜ values.
3. (A) Run two predict/update steps of a 1D constant-velocity Kalman filter by hand with given F, H, Q, R.
4. (A) Build the IoU cost matrix for 3 tracks and 3 detections and solve the assignment.
5. (B) Compute dense flow with Farnebäck (OpenCV) and RAFT on a traffic video; visualise with HSV colour coding.
6. (B) Implement SORT yourself (Kalman + Hungarian with `scipy.optimize.linear_sum_assignment`) on YOLO detections; compare with ByteTrack on an MOT17 sequence using MOTA and IDF1.
7. (C) Estimate vehicle speeds in a road video using a homography to metres and validate against a known distance between lane markings.

## Lecture 7 — 3D computer vision: stereo, structure-from-motion, SLAM foundations

**Goal:** recover depth and camera motion from images. Reading: Szeliski ch. 11–13; Hartley & Zisserman ch. 9–12; Mur-Artal et al., "ORB-SLAM" (2015); Schönberger & Frahm, "Structure-from-Motion Revisited" (COLMAP, 2016).

### 7.1 Epipolar geometry

A point x in image 1 must lie on a line (the epipolar line) in image 2. For calibrated cameras:

```latex
x_2^\top F\, x_1 = 0, \qquad E = K_2^\top F K_1 = [t]_\times R
```

F (fundamental, 7 degrees of freedom) is estimated from ≥ 8 matches (8-point algorithm with normalisation) or 7; E (essential, 5 DoF) from 5 matches, inside RANSAC (Lecture 2). Decomposing E gives the relative rotation R and translation t up to scale.

### 7.2 Stereo depth

After rectification (make epipolar lines horizontal), depth comes from disparity d between left and right images:

```latex
Z = \frac{f \cdot B}{d}
```

with focal length f (pixels) and baseline B (metres). Depth error grows with Z², so stereo is accurate only at short range. Matching: block matching, semi-global matching (SGM, used in cars), and deep stereo networks. Monocular depth networks (Depth Anything, MiDaS) predict relative depth from one image.

### 7.3 Triangulation

Given two calibrated views and a match, find the 3D point whose projections are closest to the observations (linear DLT, then refine by minimising reprojection error).

### 7.4 Structure-from-motion

Incremental SfM (COLMAP): match features across all images → initialise from a good pair (essential matrix) → add images one by one with PnP (camera pose from known 3D points) → triangulate new points → bundle adjustment. Bundle adjustment jointly optimises all cameras and points:

```latex
\min_{\{R_i, t_i\}, \{X_j\}} \sum_{i,j} \rho\Big( \big\lVert x_{ij} - \pi(K_i, R_i, t_i, X_j) \big\rVert^2 \Big)
```

with a robust loss ρ. It is a large sparse non-linear least-squares problem solved with Levenberg–Marquardt (Core 1 L5–6). Neural scene representations (NeRF, 3D Gaussian splatting) start from SfM poses to render new views.

### 7.5 SLAM foundations

Simultaneous localisation and mapping runs in real time on a moving robot or phone: track the camera frame-to-frame (visual odometry), build a map, and correct accumulated drift with loop closure (recognise a revisited place, then optimise a pose graph). Filter-based SLAM uses an EKF (Lecture 6); modern systems (ORB-SLAM3) use keyframes and bundle adjustment. Visual-inertial SLAM adds an IMU for scale and robustness. Spec 4 Robotics builds on this.

### Exercises

1. (A) For a stereo rig with f = 700 px and B = 0.12 m, compute depth at disparities 70, 7 and 1 px. What is the depth error for ±0.5 px at each?
2. (A) Show that x₂ᵀEx₁ = 0 follows from the coplanarity of the two rays and the baseline.
3. (A) Why is monocular SfM only defined up to scale? How does stereo or an IMU fix it?
4. (A) Explain why bundle adjustment's Jacobian is sparse and how that is exploited (Schur complement).
5. (B) Compute a disparity map from a Middlebury or KITTI stereo pair with OpenCV SGBM; convert to a point cloud and view it in Open3D.
6. (B) Photograph a building on campus from 30–50 viewpoints and reconstruct it with COLMAP; report reprojection error and show the sparse and dense clouds.
7. (B) Estimate relative camera pose between two photos with the 5-point algorithm and RANSAC; check against COLMAP.
8. (C) Run ORB-SLAM3 or a visual odometry pipeline on a KITTI sequence; plot the trajectory against ground truth before and after loop closure.

## Lecture 8 — Perceptive AI in practice: human pose, face recognition, behaviour analysis

**Goal:** understand people in images and video, the technical methods and the special care these systems need. Reading: Szeliski ch. 6.2, 6.5; Cao et al., "OpenPose" (2019); Deng et al., "ArcFace" (2019); Carreira & Zisserman, "Quo Vadis, Action Recognition?" (I3D, 2017).

### 8.1 Human pose estimation

Predict 2D keypoints (e.g. 17 COCO joints: nose, shoulders, elbows, …).

| Approach | How | Trade-off |
| --- | --- | --- |
| Top-down | Detect each person, then predict joint heatmaps in the crop (HRNet, ViTPose) | Accurate; cost grows with the number of people |
| Bottom-up | Detect all joints, then group them (OpenPose part-affinity fields) | Constant cost; harder grouping in crowds |
| One-stage | Predict people and keypoints together (YOLO-pose, RTMPose) | Real time |

Evaluate with OKS-based AP (object keypoint similarity, the keypoint analogue of IoU). 3D pose lifts 2D keypoints to 3D; body models (SMPL) give full meshes. Uses: sports and physiotherapy analysis, fall detection for elderly care, ergonomics in factories, sign language.

### 8.2 Face analysis pipeline

1. **Detect** faces (RetinaFace, SCRFD) and landmarks.
2. **Align** by warping landmarks to a canonical position.
3. **Embed** with a CNN trained with a margin-based loss so that the same person maps close together.
4. **Compare** embeddings with cosine similarity against a threshold: verification (1:1, "is this the same person?") or identification (1:N search in a gallery).

```latex
\mathcal{L}_{\text{ArcFace}} = -\log \frac{e^{s \cos(\theta_{y} + m)}}{e^{s \cos(\theta_{y} + m)} + \sum_{j \ne y} e^{s \cos \theta_j}}
```

The additive angular margin m forces classes apart on the hypersphere. Report the false match rate (FMR) and false non-match rate (FNMR) at a chosen threshold, per demographic group (NIST FRVT showed large differences across groups for many algorithms). Liveness / anti-spoofing detection is needed against photos and masks.

### 8.3 Action and behaviour recognition

Classify what people do in video: two-stream networks (RGB + optical flow), 3D CNNs (I3D, SlowFast), video transformers (TimeSformer, VideoMAE), or lightweight skeleton-based models (ST-GCN on pose sequences, robust to clothing and background). Temporal action localisation finds when an action starts and ends. Uses: retail analytics, sports, worker safety (missing helmet, entering danger zones), traffic violations.

### 8.4 Doing this responsibly

These systems identify and monitor people, so legal and ethical constraints are strict (Lecture 10): personal-data law (Vietnam's Decree 13/2023 treats biometric data as sensitive personal data; the EU AI Act bans or restricts several uses such as real-time remote biometric identification in public spaces). Prefer designs that need no identity (counting, pose, anonymised tracks), get consent, minimise retention, and test for group-wise error differences.

### Exercises

1. (A) Compare top-down and bottom-up pose estimation in cost for 1 vs 50 people per image.
2. (A) A face verifier has FMR 0.1% and FNMR 2% at threshold τ. What happens to both when τ increases? Which matters more for a door lock vs a bank?
3. (A) Explain why identification (1:N) has a much higher false-match risk than verification (1:1) as N grows.
4. (B) Run RTMPose or YOLO-pose on a video of an exercise; compute a joint angle (e.g. knee) over time and count repetitions.
5. (B) Build a fall detector from pose sequences on a public fall dataset (e.g. UR Fall Detection) with a simple rule and with a small GRU; compare.
6. (B) Evaluate a pretrained face-verification model on a balanced public benchmark (e.g. RFW, if accessible) and report FNMR per group at fixed FMR.
7. (C) Train a skeleton-based action classifier (ST-GCN) on a subset of NTU RGB+D or Kinetics-skeleton.

## Lecture 9 — Multi-modal perception: cameras, LiDAR, radar and depth sensors

**Goal:** combine sensors with different strengths into one reliable picture of the world, as robots and vehicles must. Reading: Szeliski ch. 13.2; Qi et al., "PointNet" (2017); Lang et al., "PointPillars" (2019); Liu et al., "BEVFusion" (2023); the nuScenes and KITTI dataset papers.

### 9.1 Sensors compared

| Sensor | Measures | Strengths | Weaknesses |
| --- | --- | --- | --- |
| RGB camera | Colour, texture | Cheap, rich semantics, reads signs | No direct depth; fails in dark, glare, rain |
| Stereo / RGB-D (structured light, ToF) | Depth per pixel | Dense depth at short range | Short range; ToF suffers in sunlight |
| LiDAR | 3D points by laser time-of-flight | Accurate geometry to 100–200 m | Expensive; sparse at distance; rain and fog |
| Radar | Range + radial velocity (Doppler) | Works in all weather; measures speed directly | Very sparse; poor shape detail |
| IMU, GNSS, wheel odometry | Motion, position | High-rate motion; global position | Drift (IMU); urban canyons (GNSS) |

No single sensor is enough in all conditions; that is the case for fusion.

### 9.2 Calibration and synchronisation

Fusion needs intrinsic calibration of each sensor, extrinsic calibration between them (the rigid transform from LiDAR to camera frame, estimated with targets or by aligning edges), and time synchronisation (hardware triggers or timestamps with motion compensation). A few centimetres or milliseconds of error visibly misaligns fused data.

### 9.3 Point-cloud processing

Classical: voxel downsampling, ground-plane removal with RANSAC (Lecture 2), Euclidean clustering, and registration with ICP (iteratively match nearest points and solve for the rigid transform with SVD, Core 1 L1). Deep learning:

- **PointNet / PointNet++:** shared MLP per point + symmetric max-pool, so the result does not depend on point order.
- **Voxel / pillar methods** (VoxelNet, PointPillars, CenterPoint): convert to grids and use 2D/3D convolutions; real time for 3D detection.
- **Projection:** range images or bird's-eye-view (BEV) maps processed as images.

### 9.4 Fusion strategies

| Level | What is fused | Example |
| --- | --- | --- |
| Early (data) | Raw data, e.g. LiDAR points projected onto the image as a depth channel | RGB-D input to a CNN |
| Middle (feature) | Learned features from each sensor in a shared space | BEVFusion: camera and LiDAR features in bird's-eye view |
| Late (decision) | Each sensor's detections, combined by association and a filter | Camera boxes + radar tracks in a Kalman filter |

Probabilistic fusion with a Kalman filter (Lecture 6) weights each sensor by its noise covariance R. Design for graceful degradation: the system should keep working, more cautiously, when one sensor fails.

### 9.5 Evaluation

3D detection uses 3D IoU or centre-distance AP (nuScenes NDS combines AP with errors in translation, scale, orientation, velocity). Always test by condition: day/night, rain, distance bands.

### Exercises

1. (A) Project a LiDAR point (5, 1, 0.5) m in LiDAR frame into an image given extrinsics \[R | t\] and K.
2. (A) Derive the closed-form rigid alignment of two matched point sets via SVD (the core of ICP).
3. (A) Fuse two independent 1D position estimates (camera: 10.0 m, σ = 1.0; radar: 10.8 m, σ = 0.3). Give the fused mean and variance.
4. (A) Explain why PointNet's max-pool makes it invariant to point order.
5. (B) Load a KITTI frame; remove the ground with RANSAC, cluster objects, and overlay LiDAR points on the camera image coloured by depth.
6. (B) Register two scans with ICP in Open3D; plot error against iterations with good and bad initialisation.
7. (B) Run a pretrained PointPillars (OpenPCDet or MMDetection3D) on KITTI; report 3D AP by distance band.
8. (C) Implement late fusion of camera detections and radar targets on a nuScenes mini scene with a Kalman filter; compare tracking with each sensor alone.

## Lecture 10 — Ethics in computer vision: fairness, privacy and bias mitigation

**Goal:** find and reduce unfair behaviour in vision systems, protect the privacy of people in images, and know the rules that apply. Reading: Buolamwini & Gebru, "Gender Shades" (2018); Grother et al., NIST FRVT Part 3: Demographic Effects (2019); Barocas, Hardt & Narayanan, *Fairness and Machine Learning* ch. 3 (free); EU AI Act summary; Vietnam Decree 13/2023/NĐ-CP on personal data protection.

### 10.1 Where bias comes from

- **Data collection:** datasets over-represent some regions, skin tones, ages and settings (ImageNet and COCO are mostly Western). Models trained on European roads may miss Vietnamese motorbike patterns and vehicles.
- **Labelling:** annotator assumptions (e.g. labelling gender from appearance), inconsistent guidelines.
- **Sensors and optics:** exposure settings tuned for lighter skin; low-light performance differences.
- **Deployment:** cameras placed mainly in some neighbourhoods; feedback loops from enforcement.

### 10.2 Measuring fairness

Evaluate per subgroup, not only on average. Common criteria for a decision ŷ and group A:

| Criterion | Requires | Example |
| --- | --- | --- |
| Demographic parity | P(ŷ = 1) equal across groups | Equal flag rates |
| Equalised odds | Equal true-positive and false-positive rates across groups | Equal error rates for a face verifier |
| Calibration within groups | Same meaning of a score in every group | Risk scores |

These generally cannot all hold at once when base rates differ, so choose the criterion that fits the harm. Report confidence intervals: small subgroups give noisy estimates.

### 10.3 Mitigation

- **Data:** collect targeted data for weak groups and conditions; rebalance; synthetic data with care.
- **Training:** group-balanced or worst-group losses (group DRO); fairness constraints.
- **Post-processing:** group-specific thresholds (legally sensitive in some contexts).
- **Process:** datasheets for datasets, model cards with subgroup results, external audits, and the option not to build a system at all.

### 10.4 Privacy-preserving perception

- Do as much as possible on the device and send only results (counts, not images).
- Blur or mask faces and licence plates at capture (Lecture 5 exercise 8).
- Keep data minimal and delete it on schedule; restrict access and log it.
- Use aggregation, federated learning or differential privacy when training on sensitive data.
- Beware re-identification: gait, clothing and location trails identify people even without faces.

### 10.5 Rules that apply

Vietnam's Decree 13/2023 requires a legal basis (usually consent) to process personal data and treats biometric data as sensitive; organisations must assess the impact of processing. The EU AI Act classifies biometric identification, critical infrastructure and some workplace monitoring as high-risk or prohibited uses, with obligations on data quality, documentation, human oversight and accuracy across groups. Many companies building for export follow both.

### Exercises

1. (A) From given per-group confusion matrices of a pedestrian detector (day vs night, or light vs dark clothing), compute TPR and FPR per group and say which fairness criterion fails.
2. (A) Show with a numeric example why equalised odds and calibration cannot both hold when base rates differ.
3. (A) Write a one-page data-protection impact assessment for a campus camera system that counts people in the canteen.
4. (B) Evaluate a pretrained person detector on images split by lighting (day/night) and by region (Western vs Vietnamese street scenes you collect); report AP per subgroup with bootstrap CIs.
5. (B) Fine-tune with extra data for the weakest subgroup and measure the change in worst-group AP.
6. (B) Write a model card for the detector you fine-tuned in Lecture 4.
7. (C) Show that a pedestrian re-identification model can match people across cameras without faces; discuss what that means for "anonymised" video.

## Project — Motorbike traffic monitoring from street cameras

**What you build:** a system that watches a Vietnamese intersection or road from a fixed camera and reports, per minute and per lane: counts by vehicle type (motorbike, car, bus, truck, bicycle), average speeds, queue length at the red light, and optional events such as riders without helmets or wrong-way driving. It is privacy-safe by design: no faces or plates are stored. City traffic centres, smart-city projects in Binh Duong and Ho Chi Minh City, and road planners use this kind of data. It uses calibration and homographies (L1), backbones and detection (L3–4), segmentation (L5), tracking and Kalman filters (L6), behaviour analysis (L8) and the ethics material (L10).

**Data:** record your own 1080p video from a safe, legal spot (a pedestrian bridge or a building overlooking a road near campus), several hours across day, night and rain. Public Vietnamese traffic image datasets (several exist from Vietnamese universities and Kaggle) help pre-train; check each licence. Label 300–500 frames with CVAT plus SAM-assisted masks, and 2–3 short clips with track IDs for evaluation.

### Stages

1. **Scene geometry (week 1):** calibrate or estimate camera intrinsics, undistort, and compute a road-plane homography from lane markings of known length (L1). Draw counting lines and lane polygons.
2. **Detection (weeks 1–2):** benchmark a COCO-pretrained YOLO and RT-DETR, then fine-tune on your labelled frames. Dense motorbike crowds stress NMS and small-object detection; try higher input resolution and tiled inference. Report mAP overall and per class, day vs night (L4).
3. **Tracking and counting (week 3):** ByteTrack or BoT-SORT; count line crossings per class and lane. Measure counting error against manual counts on 10 minutes of video, plus MOTA, IDF1 and ID switches (L6).
4. **Speed and queues (week 4):** convert tracks to metres via the homography, smooth with a Kalman filter, and compute speeds; estimate queue length from stationary tracks or a segmentation mask of the road (L5–6). Validate speed with a test ride at a known speed.
5. **Events (week 5, choose one):** helmet / no-helmet classification on rider crops, or wrong-way detection from track direction. Report precision and recall (L8).
6. **Privacy and deployment (weeks 5–6):** blur faces and plates in anything saved, keep only aggregated statistics, export to TensorRT/ONNX and measure FPS on a Jetson-class device or laptop GPU, and build a dashboard of counts and speeds over time (L10).

### Evaluation

| Metric | Meaning |
| --- | --- |
| mAP@0.5 per class, day / night / rain | Detection quality by condition |
| Counting error (%) per class vs manual counts | What a traffic planner relies on |
| MOTA, IDF1, ID switches | Tracking stability |
| Speed error (km/h) vs ground-truth test rides | Geometric accuracy |
| Event precision / recall | Usefulness of alerts |
| FPS on target hardware | Real-time feasibility |

### Report outline (10–15 pages)

1. Problem, users, site and legal/ethical setup (data-protection impact assessment)
2. Pipeline: geometry, detection, tracking, speed, events
3. Results by condition and class, with error analysis of failure cases (occlusion in dense traffic, night glare, rain)
4. Deployment measurements and dashboard
5. Limitations, privacy design, model card, AI Disclosure Statement, individual contributions

### Stretch goals

- Fuse with a cheap radar or a second camera for better speed estimates (L9).
- Estimate origin–destination flows across two cameras without identifying people (aggregate re-identification only, discuss the risks).
- Share the anonymised counts with a local authority or researcher and ask what decisions it could inform.
