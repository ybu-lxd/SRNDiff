# SRNDiff
## Abstract
Diffusion models are widely used in image generation because they can generate high-quality and realistic samples. This is in contrast to generative adversarial networks (GANs) and variational autoencoders (VAEs), which have some limitations in terms of image quality.We introduce the diffusion model to the precipitation forecasting task and propose a short-term precipitation nowcasting with condition diffusion model based on historical observational data, which is referred to as SRNDiff. By incorporating an additional conditional decoder module in the denoising process, SRNDiff achieves end-to-end conditional rainfall prediction. SRNDiff is composed of two networks: a denoising network and a conditional Encoder network. The conditional network is composed of multiple independent UNet networks. These networks extract conditional feature maps at different resolutions, providing accurate conditional information that guides the diffusion model for conditional generation.SRNDiff surpasses GANs in terms of prediction accuracy, although it requires more computational resources.The SRNDiff model exhibits higher stability and efficiency during training than GANs-based approaches, and generates high-quality precipitation distribution samples that better reflect future actual precipitation conditions. This fully validates the advantages and potential of diffusion models in precipitation forecasting, providing new insights for enhancing rainfall prediction.





# How to use



```python
python sample.py
```

The pre-weights can be accessed at the following address:
https://drive.google.com/file/d/1sZsT_Qe0_9kmXIfT8fRuZ9Rj5wY37Bbr/view?usp=drive_link



The following result will appear

![lable](sampled_images_and_label.png)

# Requirement
You only need to save torch>=0.13.1 cuda>=11.7, python>=3.7, there are no specific requirements for other packages
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```


```
torch==1.13.1
torchvision==0.14.1
tqdm==4.64.1
einops==0.6.0
matplotlib==3.5.3
numpy==1.21.6
```


# TO DO
- [x] Upload the complete inference code
- [x] Visualize the results
- [ ] Improve the training code
- [ ] Upload the attention layer visualization code
- [ ] Provide operating environment



# BibTeX
If you find this paper/project helpful, please cite us.

```
@article{ling2024srndiff,
  title={SRNDiff: Short-term Rainfall Nowcasting with Condition Diffusion Model},
  author={Ling, Xudong and Li, Chaorong and Qin, Fengqing and Yang, Peng and Huang, Yuanyuan},
  journal={arXiv preprint arXiv:2402.13737},
  year={2024}
}
```