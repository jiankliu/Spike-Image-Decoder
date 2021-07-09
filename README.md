# Spike-Image-Decoder

This is a sample code for the following work:

Zhang Y, Jia S, Zheng Y, Yu Z, Tian Y., Ma S., Huang T, Liu J.K. Reconstruction of Natural Visual Scenes from Neural Spikes with Deep Neural Networks. Neural Networks. 125:19-30 (2020) https://doi.org/10.1016/j.neunet.2020.01.033  
or the link with open access https://www.researchgate.net/publication/339127649_Reconstruction_of_natural_visual_scenes_from_neural_spikes_with_deep_neural_networks

The overall approach is part of our computational framework for visiual processing, see more:
Yu, Z., Liu, J. K., Jia, S., Zhang, Y., Zheng, J., Tian, Y., and Huang, T., Toward the Next Generation of Retinal Neuroprosthesis: Visual Computation with Spikes. Engineering. 6:449-461 (2020) https://doi.org/10.1016/j.eng.2020.02.004

In the paper, we used experimental data and simulated spikes. To generate simulated spikes, one can use any type of method in the field, such as linear-nonlinear model in neuroscience for retinal coding, or simple transcoding of image pixels into spikes by mapping pixel intensity directly in neuromorphic computing for spiking neural network.

The model is easy to implement. The core part of the model is a naive multilayer preception plus a plain autoencoder. Refer SID.py for detials.

See some demos here: 

https://sites.google.com/site/jiankliu/hightlight/visual-coding-with-spikes

