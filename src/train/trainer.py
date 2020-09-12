import os
import time
import shutil
import pickle

import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value

from src.network.dram import RecurrentAttention
from src.utils.utils import AverageMeter


class Trainer:
    """A Recurrent Attention Model trainer.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args:
            config: object containing command line arguments.
            data_loader: A data iterator.
        """
        self.config = config

        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.num_glimpses = config.num_glimpses

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.lr = config.init_lr

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.0
        self.counter = 0
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = config.model_name

        self.plot_dir = "./plots/" + self.model_name + "/"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print("[*] Saving tensorboard logs to {}".format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        # build RAM model
        self.model = RecurrentAttention(config)
        self.model.to(self.device)

        # initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.init_lr,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=config.lr_patience
        )

    def train(self):
        """Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.config.resume:
            self.load_checkpoint(best=False)

        print("\n[*] Train on {} samples, validate on {} samples"
              .format(self.num_train, self.num_valid))

        for epoch in range(self.start_epoch, self.epochs):
            print("\nEpoch: {}/{} - LR: {:.6f}".format(epoch + 1, self.epochs, self.optimizer.param_groups[0]["lr"]))

            # train for 1 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_loss, valid_acc = self.validate(epoch)

            # # reduce lr if validation loss plateaus
            self.scheduler.step(-valid_acc)

            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val err: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_loss, train_acc, valid_loss, valid_acc, 100 - valid_acc))

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint({
                "epoch": epoch + 1,
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
                "best_valid_acc": self.best_valid_acc,
            },
                is_best)

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                loss, acc, preds, locs, imgs = self.one_epoch(x, y)

                # compute gradients and update SGD
                loss.backward()
                self.optimizer.step()

                # store
                losses.update(loss.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    ("{:.1f}s - loss: {:.3f} - acc: {:.3f}".format((toc - tic), loss.item(), acc.item())))

                batch_size = x.shape[0]
                pbar.update(batch_size)

                self.__save_images_if_plotting(epoch, i, locs, imgs)

                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + i
                    log_value("train_loss", losses.avg, iteration)
                    log_value("train_acc", accs.avg, iteration)

            return losses.avg, accs.avg

    def one_epoch(self, x, y):

        # initialize location vector and hidden state
        batch_size = x.shape[0]
        x, y = x.to(self.device), y.to(self.device)

        l_t = self.model.reset(batch_size, self.device)

        imgs = []
        locs = []
        log_pi = []
        baselines = []
        for t in range(self.num_glimpses - 1):
            # forward pass through model
            h_t, l_t, b_t, p = self.model(x, l_t)

            # save locs for plotting
            locs.append(l_t[0:9])

            baselines.append(b_t)
            log_pi.append(p)

        # last iteration
        h_t, l_t, b_t, log_probas, p = self.model(x, l_t, last=True)
        log_pi.append(p)
        baselines.append(b_t)

        # save locs and images for plotting
        locs.append(l_t[0:9])
        imgs.append(x[0:9])

        # convert list to tensors and reshape
        baselines = torch.stack(baselines).transpose(1, 0)
        log_pi = torch.stack(log_pi).transpose(1, 0)

        # calculate reward
        predicted = torch.max(log_probas, 1)[1]
        R = (predicted.detach() == y).float()
        R = R.unsqueeze(1).repeat(1, self.num_glimpses)

        # compute losses for differentiable modules
        loss_action = F.nll_loss(log_probas, y)
        loss_baseline = F.mse_loss(baselines, R)

        # compute reinforce loss
        # summed over timesteps and averaged across batch
        adjusted_reward = R - baselines.detach()
        loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
        loss_reinforce = torch.mean(loss_reinforce, dim=0)

        # sum up into a hybrid loss
        loss = loss_action + loss_baseline + loss_reinforce * 0.01

        # compute accuracy
        correct = (predicted == y).float()
        acc = 100 * (correct.sum() / len(y))

        return loss, acc, predicted, locs, imgs

    def __save_images_if_plotting(self, epoch, i, locs, imgs):
        # dump the glimpses and locs
        if (epoch % self.plot_freq == 0) and (i == 0):
            imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
            locs = [l.cpu().data.numpy() for l in locs]
            pickle.dump(imgs, open(self.plot_dir + "g_{}.p".format(epoch + 1), "wb"))
            pickle.dump(locs, open(self.plot_dir + "l_{}.p".format(epoch + 1), "wb"))

    @torch.no_grad()
    def validate(self, epoch):
        """Evaluate the RAM model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            x, y = x.to(self.device), y.to(self.device)

            # initialize location vector and hidden state
            batch_size = x.shape[0]
            l_t = self.model.reset(batch_size, self.device)

            # extract the glimpses
            log_pi = []
            baselines = []
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t)

                # store
                baselines.append(b_t)
                log_pi.append(p)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(x, l_t, last=True)
            log_pi.append(p)
            baselines.append(b_t)

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            # average
            # M
            log_probas = log_probas.view(1, -1, log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)

            # M
            baselines = baselines.contiguous().view(1, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)

            # M
            log_pi = log_pi.contiguous().view(1, -1, log_pi.shape[-1])
            log_pi = torch.mean(log_pi, dim=0)

            # calculate reward
            predicted = torch.max(log_probas, 1)[1]
            R = (predicted.detach() == y).float()
            R = R.unsqueeze(1).repeat(1, self.num_glimpses)

            # compute losses for differentiable modules
            loss_action = F.nll_loss(log_probas, y)
            loss_baseline = F.mse_loss(baselines, R)

            # compute reinforce loss
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)

            # sum up into a hybrid loss
            loss = loss_action + loss_baseline + loss_reinforce * 0.01

            # compute accuracy
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.item(), x.size()[0])
            accs.update(acc.item(), x.size()[0])

            # log to tensorboard
            if self.use_tensorboard:
                iteration = epoch * len(self.valid_loader) + i
                log_value("valid_loss", losses.avg, iteration)
                log_value("valid_acc", accs.avg, iteration)

        return losses.avg, accs.avg

    @torch.no_grad()
    def test(self):
        """Test the RAM model.

        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0
        preds = []
        # load the best checkpoint
        self.load_checkpoint(best=self.best)

        for i, (x, y) in enumerate(self.test_loader):
            x, y = x.to(self.device), y.to(self.device)

            # initialize location vector and hidden state
            batch_size = x.shape[0]
            l_t = self.model.reset(batch_size, self.device)

            # extract the glimpses
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(x, l_t, last=True)

            # M
            log_probas = log_probas.view(1, -1, log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)

            pred = log_probas.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            preds.append(pred)
        perc = (100.0 * correct) / (self.num_test)
        error = 100 - perc
        print(
            "[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)".format(
                correct, self.num_test, perc, error
            )
        )
        return preds

    def save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.

        If this model has reached the best validation accuracy thus
        far, a separate file with the suffix `best` is created.
        """
        filename = self.model_name + "_ckpt.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    def load_checkpoint(self, best=False):
        """Load the best copy of a model.
        Args:
            best: if set to True, loads the best model.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # load variables from checkpoint
        self.start_epoch = ckpt["epoch"]
        self.best_valid_acc = ckpt["best_valid_acc"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt["epoch"], ckpt["best_valid_acc"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"]))
