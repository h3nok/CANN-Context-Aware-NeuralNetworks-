# Deep-CLO : Deep Curriculum Learning Optimization

### Abstract
We describe a quantitative and practical framework to integrate curriculum learning (CL) into deep 
learning training pipeline to improve feature learning in deep feed-forward networks. The framework 
has several unique characteristics: (1) dynamicity—it proposes a set of batch-level training strategies 
(syllabi or curricula) that are sensitive to data complexity (2) adaptivity—it dynamically estimates 
the effectiveness of a given strategy and performs objective comparison with alternative strategies 
making the method suitable both for practical and research purposes. (3) Employs replace–retrain 
mechanism when a strategy is unfit to the task at hand. In addition to these traits, the framework can
combine CL with several variants of gradient descent (GD) algorithms and has been used to generate 
efficient batch-specific or data-set specific strategies. Comparative studies of various current 
state-of-the-art vision models, such as FixEfficentNet and BiT-L (ResNet), on several benchmark datasets
including CIFAR10 demonstrate the effectiveness of the proposed method. We present results that show 
training loss reduction by as much as a factor 5. Additionally, we present a set of practical curriculum
strategies to improve the generalization performance of select networks on various datasets.

![img.png](deep_clo_framework.png)

https://link.springer.com/article/10.1007/s42979-020-00251-7

