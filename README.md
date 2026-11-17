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
