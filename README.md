# keras-
用keras搭建的CNN神经网络，来识别花朵

flower-fine.py通过ResNet50，VGG19，InceptionV3进行特征提取，主要是通过以训练好的cnn层提取特征

make_data.py将提取的特征制作成训练集，测试集训练我们需要的网络

train.py网络训练

make_parallel.py多GPU训练，因为我的配置是2个1080Ti的显卡，需要配置这个，使用单个gpu或者cpu测不需要

