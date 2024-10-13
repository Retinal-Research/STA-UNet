# STA-Unet: Rethink the semantic redundant for Medical Imaging Segmentation (under review in WACV 2025)
In recent years, significant progress has been made in the medical image analysis domain using convolutional neural networks (CNNs). In particular, deep neural networks based on a U-shaped architecture (UNet) with skip connections have been adopted for several medical imaging tasks, including organ segmentation. Despite their great success, CNNs are not good at learning global or semantic features. Especially ones that require human-like reasoning to understand the context. Many UNet architectures attempted to adjust with the introduction of Transformer-based self-attention mechanisms, and notable gains in performance have been noted. However, the transformers are inherently flawed with redundancy to learn at shallow layers, which often leads to an increase in the computation of attention from the nearby pixels offering limited information. The recently introduced Super Token Attention (STA) mechanism adapts the concept of superpixels from pixel space to token space, using super tokens as compact visual representations. This approach tackles the redundancy by learning efficient global representations in vision transformers, especially for the shallow layers. In this work, we introduce the STA module in the UNet architecture (STA-UNet), to limit redundancy without losing rich information. Experimental results on four publicly available datasets demonstrate the superiority of STA-UNet over existing state-of-the-art architectures in terms of Dice score and IOU for organ segmentation tasks. 

## Model Architecture




## Training on Synapse Dataset 


