# STA-Unet: Rethink the semantic redundant for Medical Imaging Segmentation (under review in WACV 2025)
In recent years, significant progress has been made in the medical image analysis domain using convolutional neural networks (CNNs). In particular, deep neural networks based on a U-shaped architecture (UNet) with skip connections have been adopted for several medical imaging tasks, including organ segmentation. Despite their great success, CNNs are not good at learning global or semantic features. Especially ones that require human-like reasoning to understand the context. Many UNet architectures attempted to adjust with the introduction of Transformer-based self-attention mechanisms, and notable gains in performance have been noted. However, the transformers are inherently flawed with redundancy to learn at shallow layers, which often leads to an increase in the computation of attention from the nearby pixels offering limited information. The recently introduced Super Token Attention (STA) mechanism adapts the concept of superpixels from pixel space to token space, using super tokens as compact visual representations. This approach tackles the redundancy by learning efficient global representations in vision transformers, especially for the shallow layers. In this work, we introduce the STA module in the UNet architecture (STA-UNet), to limit redundancy without losing rich information. Experimental results on four publicly available datasets demonstrate the superiority of STA-UNet over existing state-of-the-art architectures in terms of Dice score and IOU for organ segmentation tasks. 

## Model Architecture

![Model Overview](https://github.com/Retinal-Research/STA-UNet/blob/master/images/architecture%20illustration.png)

## Visual Comparison with existing methods. 

![Results](https://github.com/Retinal-Research/STA-UNet/blob/master/images/synapse%20illustration.png)

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

- [Git](https://git-scm.com)
- [Python](https://www.python.org/downloads/) and [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) (optional)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Retinal-Research/STA-UNet.git

2. Create a Python Environment and install the required libraries by running
   ```sh
   pip install -r requirements.txt
   
## Downloading Synapse Dataset 

The datasets we used are provided by TransUnet's authors. The preprocessed Synapse dataset is accessed from [here](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd). extract the zip file and copy the **data** folder to your project directory.

## Inference on Synapse Dataset
model_weight
The pre-trained weights on Synapse Dataset can be downloaded from [here](https://drive.google.com/drive/folders/1hjoffuESP3bAnV_SSGmkvhHwjtj8DWvz?usp=sharing). After extracting the weights file to your **OUTPUT_PATH** (and downloading the Synapse dataset). You can run the following command in terminal to infer on the testing set using proposed STA-UNet. 

```sh
python test.py --output_dir OUTPUT_PATH --max_epochs 150
