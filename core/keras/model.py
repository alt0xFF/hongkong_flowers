import os
import tensorflow as tf
from keras import backend as K

from core.keras.parser import ModelsDict, OptimsDict, LossesDict, MetricsDict, TransformsDict


class FlowerClassificationModel(object):
    def __init__(self, args):
        super(self.__class__, self).__init__()

        # params
        self.train_dir = args.data_dir + 'train/'
        self.valid_dir = args.data_dir + 'valid/'
        self.test_dir = args.data_dir + 'test/'
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs

        self.width = args.width
        self.height = args.height

        # this setting allocates memory dynamically
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
        K.set_session(sess)

        # setup model
        model_choice, opt_choice, loss_choice, metric_choice = args.configs

        # remember these are classes
        Optim = OptimsDict[opt_choice]
        Loss = LossesDict[loss_choice]

        # for keras, we use functional flowermodel!
        FlowerModel = ModelsDict[model_choice]

        # get folder names in train_dir, note that this is str!
        self.classes = [i for i in os.listdir(self.train_dir) if os.path.isdir(self.train_dir + i)]

        # count folders in data_dir
        self.num_classes = len(self.classes)

        # transform is an initialized ImageDataGenerator object for keras
        self.transform = TransformsDict[args.transform]

        # metric is a list of function!
        self.metric = MetricsDict[metric_choice]

        # initialize model
        self.model = FlowerModel(args, self.num_classes)

        # set up loss function, this is useless here but for the sake of completeness I leave it here.
        self.criterion = Loss

        # set up optimizer
        self.optimizer = Optim(lr=0.001)

        # compile model
        self.model.compile(loss=self.criterion, optimizer=self.optimizer, metrics=self.metric)

    def fit(self):

        # datasets and dataloader for training
        train_generator = self.transform.flow_from_directory(self.train_dir,
                                                             batch_size=self.batch_size,
                                                             target_size=(self.height, self.width),
                                                             classes=self.classes,
                                                             class_mode='categorical')

        valid_generator = self.transform.flow_from_directory(self.valid_dir,
                                                             batch_size=self.batch_size,
                                                             target_size=(self.height, self.width),
                                                             classes=self.classes,
                                                             class_mode='categorical')

        train_step = train_generator.samples // self.batch_size
        valid_step = valid_generator.samples // self.batch_size

        # fit the model
        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=train_step,
                                 epochs=self.num_epochs,
                                 validation_data=valid_generator,
                                 validation_steps=valid_step)

    def evaluate(self):

        # datasets and dataloader for test mode
        test_generator = self.transform.flow_from_directory(self.test_dir,
                                                            batch_size=self.batch_size,
                                                            target_size=(self.height, self.width),
                                                            classes=self.classes)

        test_step = test_generator.samples // self.batch_size

        loss = self.model.evaluate_generator(generator=test_generator,
                                             steps=test_step)

        print('Test set loss = %.4f' % loss)

    def predict(self, x):
        self.model.predict(x)


