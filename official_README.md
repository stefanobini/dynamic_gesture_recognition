# Benchmarking CNNs for gesture recognition on embedded devices equipped by robots
```diff
- Note: Code and models will be added after the acceptance of our eponymous work.
```

The gesture is one of the most used forms of communication between humans; in recent years, given the new trend of factories to be adapted to Industry 4.0 paradigm, the scientific community has shown a growing interest towards the design of Gesture Recognition (GR) algorithms for Human-Robot Interaction (HRI) applications.
Within this context, the GR algorithm needs to work in real time and over embedded platforms, with limited resources. Anyway, when looking at the available scientific literature, the aim of the different proposed neural networks (i.e. 2D and 3D) and of the different modalities used for feeding the network (i.e. RGB, RGB-D, optical flow) is typically the optimization of the accuracy, without strongly paying attention to the feasibility over low power hardware devices. 
Anyway, the analysis related to the trade-off between accuracy and computational burden (for both networks and modalities) becomes important so as to allow GR algorithms to work in industrial robotics applications. 
We perform a wide benchmarking analysis focusing not only on the accuracy but also on the computational burden, involving two different architectures (2D and 3D), with two different backbones (MobileNet, ResNeXt) and four types of input modalities (RGB, Depth, Optical Flow, Motion History Image) and their combinations.

Questo framework, realizzato in PyTorch, continene il codice utilizzato per effettuare questa analisi, descritta in [Benchmarking CNNs for gesture recognition on embedded devices equipped by robots](link).

![alt text](https://github.com/stefanobini/gesture_recognition/blob/main/workflow.png)
