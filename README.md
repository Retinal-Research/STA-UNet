# STA-Unet: Rethink the semantic redundant for Medical Imaging Segmentation (under review in WACV 2025)
In recent years, significant progress has been made in the
medical image analysis domain using convolutional neu-
ral networks (CNNs). In particular, deep neural networks
based on a U-shaped architecture (UNet) with skip connec-
tions have been adopted for several medical imaging tasks,
including organ segmentation. Despite their great success,
CNNs are not good at learning global or semantic features.
Especially ones that require human-like reasoning to un-
derstand the context. Many UNet architectures attempted
to adjust with the introduction of Transformer-based self-
attention mechanisms, and notable gains in performance
have been noted. However, the transformers are inherently
flawed with redundancy to learn at shallow layers, which
often leads to an increase in the computation of attention
from the nearby pixels offering limited information. The re-
cently introduced Super Token Attention (STA) mechanism
adapts the concept of superpixels from pixel space to to-
ken space, using super tokens as compact visual represen-
tations. This approach tackles the redundancy by learning
efficient global representations in vision transformers, es-
pecially for the shallow layers. In this work, we introduce
the STA module in the UNet architecture (STA-UNet), to
limit redundancy without losing rich information. Experi-
mental results on four publicly available datasets demon-
strate the superiority of STA-UNet over existing state-of-
the-art architectures in terms of Dice score and IOU for or-
gan segmentation tasks

