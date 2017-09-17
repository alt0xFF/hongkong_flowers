from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from core.pytorch.dataset import FlowerDataset
from core.pytorch.parser import ModelsDict, OptimsDict, LossesDict, MetricsDict, TransformsDict
from core.pytorch.utils import AverageMeter

class Model(object):
    def __init__(self, args):
        super(Model, self).__init__()
        
        # this is for removing torch dependency in options.py
        if args.use_cuda: 
            args.use_cuda = torch.cuda.is_available()
        
        # set up visdom
        if args.visualize:
            self.vis = visdom.Visdom()
            
        # parsed transform function for dataset
        transform = TransformsDict[args.transform]
        
        # datasets and dataloader for train mode
        if args.mode == 0:
            self.train_dataset = FlowerDataset(args.data_dir, 'train', transform)
            self.valid_dataset = FlowerDataset(args.data_dir, 'valid', transform)

            self.train_dataloader = DataLoader(self.train_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=True)
            self.valid_dataloader = DataLoader(self.valid_dataset, 
                                               batch_size=args.valid_batch_size, 
                                               shuffle=False)
            
        # datasets and dataloader for test mode            
        if args.mode == 1:
            self.test_dataset = FlowerDataset(args.data_dir, 'test', transform)
            self.test_dataloader = DataLoader(self.test_dataset, 
                                              batch_size=args.test_batch_size, 
                                              shuffle=False)
                
        # setup model
        model_choice, opt_choice, loss_choice, metric_choice = args.configs
        
        # remember these are classes
        FlowerModel = ModelsDict[model_choice]
        Optim = OptimsDict[opt_choice]
        Loss = LossesDict[loss_choice]
        
        # but metric is a function!
        self.metric = MetricsDict[metric_choice]
        
        # get number of classes from dataset
        num_classes = self.train_dataset.num_classes
        
        # initialize model
        self.model = FlowerModel(args, num_classes) 

        # set up data parallel and cuda for model
        if args.use_cuda:
            if args.dataparallel:
                self.model = torch.nn.DataParallel(self.model).cuda()
            else:
                self.model = self.model.cuda()         
        
        self.criterion = Loss()   
        self.optimizer = Optim(self.model.parameters(), lr=args.lr)

    def train(self, train_dataloader, model, criterion, optimizer, epoch, args): 
    
        # switch to train mode
        model.train()

        end = time.time() # this one is for each update, will be updated every print interval.
        start_time = time.time() # this one is for each epoch. 

        # logs for loss, grad_norm and time
        losses = AverageMeter()
        grad_norm = AverageMeter()
        load_time = AverageMeter()
        total_time = AverageMeter()

        for i_batch, batch in enumerate(train_dataloader):

            load_time.update(time.time() - end) # logs the time taken for loading data

            inputs = Variable(batch["image"], requires_grad=False)
            labels = Variable(batch["label"], requires_grad=False)
            labels = torch.squeeze(labels)

            if args.use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # forward pass
            outputs = model(inputs) # batch size * num_class

            # calculate softmax loss
            loss = criterion(outputs, labels) 
            loss_sum = loss.data[0]
            losses.update(loss_sum)

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            grad_norm.update(torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_clip_norm))

            # update weights
            optimizer.step()

            if args.use_cuda:
                torch.cuda.synchronize()

            if i_batch % args.log_interval == args.log_interval-1:          
                sys.stdout.write("\r[%d, %5d] loss: %.3f, clipped grad norm: %.3f, load: %.2fs, total: %.2fs" % 
                      (epoch+1, i_batch+1, losses.avg, grad_norm.avg, load_time.avg * args.log_interval, total_time.avg * args.log_interval))
                sys.stdout.flush()
                grad_norm.reset() # reset here so only print the avg in print interval
                load_time.reset()
                total_time.reset()

            total_time.update(time.time() - end)

            # end time        
            end = time.time()

        print("\n\nEpoch %i: Train Loss = %.4f, Total Time = %.4fs" % (epoch, losses.avg, time.time() - start_time))
    
    
    def validate(self, val_dataloader, model, criterion, epoch, args):
        # switch to eval mode
        model.eval()

        start_time = time.time()

        losses = AverageMeter()
        total_acc = AverageMeter()    

        for i_batch, batch in enumerate(val_dataloader):

            inputs = Variable(batch["image"], requires_grad=False)
            batch_size = inputs.size(0)
            labels = Variable(batch["label"], requires_grad=False)
            labels = torch.squeeze(labels)

            if args.use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)

            # calculate accuracy
            accuracy = self.metric(outputs, labels)

            # calculate softmax loss
            loss = criterion(outputs, labels) 
            loss = loss.data[0]

            losses.update(loss)
            total_acc.update(accuracy)

            if args.use_cuda:
                torch.cuda.synchronize()

        print("Epoch %i: Valid Accuracy = %.4f, Valid Loss = %.4f, Total Time = %.4fs \n" % (epoch,total_acc.avg, losses.avg, time.time() - start_time))

    def fit(self, args):
        for epoch in range(args.num_epochs):
            print("---------------------------------------------------------")
            print('Epoch %d:' % epoch)

            # train for one epoch
            self.train(self.train_dataloader, self.model, self.criterion, self.optimizer, epoch, args)

            # evaluate on validation set after 1 epoch
            self.validate(self.valid_dataloader, self.model, self.criterion, epoch, args)
    
    # TODO: test function    
    def test(self, args):
        pass