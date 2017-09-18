import os
from tensorflow.contrib.keras.python.keras.layers import Input

from core.keras.parser import ModelsDict, OptimsDict, LossesDict, MetricsDict, TransformsDict

class Model(object):
    def __init__(self, args):
        super(Model, self).__init__()
        
        self.train_dir = args.data_dir + 'train/'
        self.valid_dir = args.data_dir + 'valid/'        
        self.test_dir = args.data_dir + 'test/'        
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        
        # get folder names in train_dir, note that this is str!
        self.classes = [i for i in os.listdir(self.train_dir) if os.path.isdir(self.train_dir + i)]
                
        # count folders in data_dir
        self.num_classes = len(self.classes)        
        
        # transform is an initialized ImageDataGenerator object for keras
        transform = TransformsDict[args.transform]
        
        # datasets and dataloader for train mode
        if args.mode == 0:
            self.train_generator = transform.flow_from_directory(self.train_dir,
                                                                 batch_size=self.batch_size,
                                                                 target_size=(args.height, args.width),
                                                                 classes=self.classes,
                                                                 shuffle=True)
            
            self.valid_generator = transform.flow_from_directory(self.valid_dir,
                                                                 batch_size=self.batch_size,
                                                                 target_size=(args.height, args.width),
                                                                 classes=self.classes)
            
            self.train_step = self.train_generator.samples // args.batch_size            
            self.valid_step = self.valid_generator.samples // args.batch_size

        # datasets and dataloader for test mode            
        if args.mode == 1:
            self.test_generator = transform.flow_from_directory(self.test_dir,
                                                                batch_size=self.batch_size,
                                                                target_size=args.img_size,
                                                                classes=self.classes)
            
            self.test_step = self.test_generator.samples // args.batch_size
            
        # setup model
        model_choice, opt_choice, loss_choice, metric_choice = args.configs
        
        # remember these are classes
        Optim = OptimsDict[opt_choice]
        Loss = LossesDict[loss_choice]
        
        # for keras, we use functional flowermodel!
        FlowerModel = ModelsDict[model_choice]
        
        # metric is also a function!
        self.metric = MetricsDict[metric_choice]
        
        # Input
        inputs = Input(name='the_input', shape=args.img_size, dtype='float32')
        
        # initialize model
        self.model = FlowerModel(args, inputs, self.num_classes) 
        
        # set up loss function, this is useless here but for the sake of completeness I leave it here.
        self.criterion = Loss  
        
        # set up optimizer
        self.optimizer = Optim(lr=args.lr)
        
        # compile model
        self.model.compile(loss=self.criterion, optimizer=self.optimizer, metrics=[self.metric])

    def fit(self):
        # fit the model
        self.model.fit_generator(generator=self.train_generator, 
                                 steps_per_epoch=self.train_step,
                                 epochs=self.num_epochs,
                                 validation_data=self.valid_generator,
                                 validation_steps=self.valid_step)
    
    # TODO: test function    
    def test(self, args):
        pass