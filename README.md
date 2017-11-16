# Hong Kong Flower Identitfication APP
An APP for identify flowers in Hong Kong with Deep Learning technology.

- Flower list provided by [柴娃娃植物網](https://www.facebook.com/groups/cwwHKplant/)
- Image annotated by [Hong Kong Deep Learning](https://www.facebook.com/groups/170776840085989/)

![screen.jpg](/images/screen.jpg)

Prerequisite
============
Keras or pyTorch installed.

Docker
======
It is very easy to set up a docker container for pyTorch and Keras using the following command.

### pyTorch:
`sudo docker pull floydhub/pytorch:0.2.0-gpu-py2.11`

(Note: remove `-gpu` if you want CPU only. Change to `py3.11` if using python3.)

`sudo nvidia-docker run -ti -v YOURDIRECTORY:/workspace/ -p 8889:8888 -p 8097:8097 floydhub/pytorch:0.2.0-gpu-py2.11`

For CPU use normal docker. You can also add `/bin/bash` at the end of the command to use bash instead.

Go to your `localhost:8889` to access pyTorch Jupyter notebook!

### Keras
`sudo docker pull floydhub/tensorflow:1.3.0-gpu-py2_aws.12` (settings likewise as above.)

`sudo nvidia-docker run -ti -v YOURDIRECTORY:/workspace/ -p 8888:8888 -p 6006:6006 floydhub/tensorflow:1.3.0-gpu-py2_aws.12`

Go to your `localhost:8888` to access Keras and Tensorflow Jupyter notebook!

Fine-tuning
===========
Fine-tuning the pre-trained ResNet50 with Oxford 102 flowers dataset

`./finetuning/boostrap.sh` to download oxford102 dataset

`python resnet50.py` to start fine-tuning

Training
========
Go to `options.py` and change data_dir to your own dataset *ABSOLUTE* path.

Choose which library, model, optimizer and loss to run in `options.py` by changing `self.configs`.

`python train.py` to start training.

Adding your own model
=====================
To add your model, simply do the following:
1. create your model class in `core/YOURLIBRARYCHOICE/models`, note that it must take two arguments `(args, num_classes)`.
2. add your model class to `ModelsDict` in `core/YOURLIBRARYCHOICE/parser.py`
3. add your model, optimizer and loss function of your choice to `CONFIGS` in `options.py`
4. change `self.configs` to your model in `options.py`

## TODO
### pyTorch:
- [ ] Visdom.
- [ ] logger for saving results.
- [ ] model.test()

### Keras:
- [x] Model.
- [x] dataloading.
- [x] model.train()
- [x] model.validate()
- [ ] model.test()
- [ ] tensorboard.

Citations:
==========
[Fine-tuning Deep Convolutional Networks for Plant Recognition](http://ceur-ws.org/Vol-1391/121-CR.pdf)
