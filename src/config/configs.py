from datasets.datasets import DatasetName
class Config():
    def __init__(self):
        # glimpse network params
        self.patch_size      = 8         # size of extracted patch at highest res
        self.glimpse_scale   = 8         # scale of successive patches
        self.num_patches     = 2         # Num of downscaled patches per glimpse
        self.loc_hidden      = 128       # hidden size of loc fc layer
        self.glimpse_hidden  = 128       # hidden size of glimpse fc

        # core network params
        self.num_glimpses    = 6         # Num of glimpses, i.e. BPTT iterations
        self.hidden_size     = 256       # hidden size of rnn

        # reinforce params
        self.std             = 0.11      # gaussian policy standard deviation
        self.M               = 1         # Monte Carlo sampling for valid and test sets

        # action network
        self.num_classes     = 4         # the number of classes

        # ETC params
        self.valid_size      = 0.1       # Proportion of training set used for validation
        self.batch_size      = 25       # Num of images in each batch of data
        self.num_workers     = 4         # Num of subprocesses to use for data loading
        self.shuffle         = True      # Whether to shuffle the train and valid indices
        self.show_sample     = False     # Whether to visualize a sample grid of the data

        # training params
        self.is_train        = True      # Whether to train(true) or test the model
        self.resume          = False     # Whether to resume training from checkpoint
        self.weight_decay    = 1e-5      # Weight decay for regularization
        self.momentum        = 0.5       # Nesterov momentum value TODO not used
        self.epochs          = 1000       # Num of epochs to train for
        self.init_lr         = 0.01      # Initial learning rate value
        self.lr_patience     = 50        # Number of epochs to wait before reducing lr
        self.train_patience  = 100       # Number of epochs to wait before stopping train

        # other params
        self.use_gpu         = True      # Whether to run on the GPU
        self.best            = True      # Load best model or most recent for testing
        self.random_seed     = 1         # Seed to ensure reproducibility
        self.data_dir        = "./data"  # Directory in which data is stored
        self.ckpt_dir        = "./ckpt"  # Directory in which to save model checkpoints
        self.logs_dir        = "./logs/" # Directory in which Tensorboard logs wil be stored
        self.use_tensorboard = False     # Whether to use tensorboard for visualization
        self.print_freq      = 100       # How frequently to print training details
        self.plot_freq       = 1         # How frequently to plot glimpses
        self.dataset         = DatasetName.CLOSED_SQUARES
        self.model_name      = "ram_{}_{}x{}_{}".format(
            self.num_glimpses,
            self.patch_size,
            self.patch_size,
            self.glimpse_scale,
        )