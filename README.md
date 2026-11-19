# BASE PIPELINE

In this work I'm trying to implement the basic pipeline in order to see how different KFS methods influence the responses of a VLM.

conda activate base_pipeline

python src/main.py

## ENVIRONMENT

I'm using Python 3.10, with a Virtual Environment.

In the requirements.txt file you can check the libraries used in this project.

OS: Fedora Linux 43 (Workstation Edition) x86_64
CPU: AMD Ryzen 5 3600 (12) @ 4.05 GHz
GPU: NVIDIA GeForce GTX 980 Ti
Memory: 23.40 GiB

I used this website to check for PyTorch compatibility with my GPU: [GitHub](https://racinmat.github.io/pytorch-compatibility/)

## PRELIMINARY RESULTS

### VSI-BENCH EVALUATION SUMMARY (10 VIDEOS)

Method: Fourier (Freq Domain)            | Avg Frames: 3078.3 | Overall Accuracy: 16.55%
Method: Pixel/Intensity (Smoothed SAD)   | Avg Frames: 27.0 | Overall Accuracy: 17.24%
Method: Spatial HSV Histogram            | Avg Frames: 21.0 | Overall Accuracy: 14.94%
Method: Local Descriptors (ORB)          | Avg Frames: 448.1 | Overall Accuracy: 13.79%
Method: Optical Flow (Sparse LK)         | Avg Frames: 533.3 | Overall Accuracy: 13.79%
Method: Optical Flow (Dense Farneback)   | Avg Frames: 151.0 | Overall Accuracy: 15.17%
Method: Stacked Cascade + EgoMotion      | Avg Frames: 47.6 | Overall Accuracy: 17.24%


### VSI-BENCH (DEBIASED) TWO-STAGE SUMMARY — 10 VIDEOS (CLIP Top-6 -> BLIP-VQA)

KFS Method                       | Avg KFs | Keep % | Sharpness | ms/frame |  VLM Acc
------------------------------------------------------------------------------------------------------------
Uniform Baseline (step=30)       |   160.5 |   3.3% |     172.0 |     0.57 |   18.34%
Fourier (Freq Domain)            |    37.9 |   0.9% |     163.8 |     3.96 |   15.43%
Pixel/Intensity (Smoothed SAD)   |   388.9 |   9.0% |     160.9 |     1.02 |   15.94%
Spatial HSV Histogram            |   540.8 |  12.5% |     133.7 |     1.70 |   16.11%
Local Descriptors (ORB)          |   736.0 |  16.5% |     141.8 |     5.61 |   15.71%
Optical Flow (Sparse LK)         |   573.8 |  12.8% |     137.9 |     2.38 |   16.17%
Optical Flow (Dense Farneback)   |   326.7 |   7.3% |     148.6 |     4.21 |   14.29%
Stacked Cascade + EgoMotion      |   313.5 |   7.0% |     153.7 |     4.86 |   15.09%