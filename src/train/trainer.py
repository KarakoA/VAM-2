import logging
import os
import time
import shutil
import pickle

import torch
import torch.nn.functional as F
from torch.distributions import Normal

import itertools

from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value

from network.dram import RecurrentAttention
from utils.utils import AverageMeter

import numpy as np
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
        self.best_valid_acc = 0.0
        self.counter = 0

        self.plot_dir = "./plots/" + self.config.model_name + "/"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # configure tensorboard logging
        if self.config.use_tensorboard:
            tensorboard_dir = self.config.logs_dir + self.config.model_name
            logging.info("[*] Saving tensorboard logs to {}".format(tensorboard_dir))
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

        logging.info("\n[*] Train on {} samples, validate on {} samples"
              .format(self.num_train, self.num_valid))

        for epoch in range(self.start_epoch, self.epochs):
            logging.info("\nEpoch: {}/{} - LR: {:.6f}".format(epoch + 1, self.epochs, self.optimizer.param_groups[0]["lr"]))

            # train for 1 epoch
            train_loss, train_acc, loss_act,loss_base,loss_reinf = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_loss, valid_acc = self.validate(epoch)

            # # reduce lr if validation loss plateaus
            self.scheduler.step(-valid_acc)

            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} - action: {:.3f}, baseline: {:.3f} reinforce: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            logging.info(msg.format(train_loss, train_acc, loss_act,loss_base,loss_reinf , valid_loss, valid_acc))

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.config.train_patience:
                logging.info("[!] No improvement in a while, stopping training.")
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
        losses_action = AverageMeter()
        losses_reinforce = AverageMeter()
        losses_baseline = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):

                loss, acc, preds, locs, imgs, loss_action ,loss_baseline, loss_reinforce  = self.one_batch(x, y)

                self.optimizer.zero_grad()

                # compute gradients and update SGD
                loss.backward()
                self.optimizer.step()

                # store
                losses.update(loss.item(), x.size()[0])
                losses_reinforce.update(loss_reinforce.item(), x.size()[0])
                losses_baseline.update(loss_baseline.item(), x.size()[0])
                losses_action.update(loss_action.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    ("{:.1f}s - loss: {:.3f} - acc: {:.3f}".format((toc - tic), loss.item(), acc.item())))

                batch_size = x.shape[0]
                pbar.update(batch_size)

                # log to tensorboard
                if self.config.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + i
                    log_value("train_loss", losses.avg, iteration)
                    log_value("train_acc", accs.avg, iteration)

            return losses.avg, accs.avg, losses_action.avg,losses_baseline.avg,losses_reinforce.avg

    def one_batch(self, x, y):
        # initialize location vector and hidden state
        batch_size = x.shape[0]
        x, y = x.to(self.device), y.to(self.device)

        imgs = []
        locs = []
        means = []
        baselines = []
        locations = []
        l_t = self.model.reset(batch_size, self.device)
        locs.append(l_t[0:9])
        for t in range(self.num_glimpses - 1):
            # forward pass through model
            h_t, l_t, b_t, mean_t = self.model(x, l_t)

            # save locs for plotting
            locs.append(l_t[0:9])
            locations.append(l_t)
            baselines.append(b_t)
            means.append(mean_t)

        # last iteration
        _, _, _, probabilities, _ = self.model(x, l_t, last=True)

        # save locs and images for plotting
        imgs.append(x[0:9])

        # convert list to tensors and reshape
        baselines = torch.stack(baselines).transpose(1, 0)
        means = torch.stack(means).transpose(1, 0)
        locations = torch.stack(locations).transpose(1, 0)

        # calculate reward
        predicted = torch.argmax(probabilities, 1)
        R = (predicted.detach() == y).float()
        #print(f"Act:  {np.bincount(y.numpy())}")
        #print(f"Pred: {np.bincount(predicted.numpy())}")
        #print(f"Base: {baselines.sum(dim=0)}")
        #print(f"R:     {R.sum()}")
        #print("---------------")
        # either 1 (if correct) or 0
        R = R.unsqueeze(1).repeat(1, self.num_glimpses-1)

        # compute losses for differentiable modules
        # smaller, better, no need invert for nll
        loss_action = F.nll_loss(probabilities, y)

        loss_baseline = F.mse_loss(baselines, R)

        # compute reinforce loss

        adjusted_reward = R - baselines.detach()

        adjusted_reward=adjusted_reward.repeat(1, 2).reshape(self.config.batch_size,-1,2).detach()
        probs = Normal(means, self.model.locator.std).log_prob(locations)
        # summed over timesteps and averaged across batch
        loss_reinforce = torch.sum(-probs * adjusted_reward, dim=1).sum(dim = 1)
        loss_reinforce = torch.mean(loss_reinforce, dim=0)

        loss = loss_action + loss_baseline + loss_reinforce * self.config.reward_multi

        # compute accuracy
        correct = (predicted == y).float()
        acc = 100 * (correct.sum() / len(y))

        return loss, acc, predicted, locs, imgs, loss_action ,loss_baseline, loss_reinforce

    def __save_images_if_plotting(self, epoch, i, locs, imgs,y):
        # dump the glimpses and locs
        if (epoch % self.config.plot_freq == 0) and (i == 0):
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
        self.model.eval()

        for i, (x, y) in enumerate(self.valid_loader):
            # 3, 3, 0, 2, 3, 0, 1, 1, 1
            loss, acc, preds, locs, imgs, _,_,_ = self.one_batch(x, y)
            self.__save_images_if_plotting(epoch, i, locs, imgs,y)
            # store
            losses.update(loss.item(), x.size()[0])
            accs.update(acc.item(), x.size()[0])

            # log to tensorboard
            if self.config.use_tensorboard:
                iteration = epoch * len(self.valid_loader) + i
                log_value("valid_loss", losses.avg, iteration)
                log_value("valid_acc", accs.avg, iteration)

        return losses.avg, accs.avg

    @torch.no_grad()
    def test(self):
        """
        Test the RAM model.
        """
        correct = 0
        preds = []

        # load the best checkpoint
        self.load_checkpoint(best=self.best)
        self.model.eval()

        for i, (x, y) in enumerate(self.test_loader):
            loss, acc, predictions, locs, imgs,_,_,_ = self.one_batch(x, y)

            correct += sum(predictions == y)
            preds.append(predictions)
        perc = (100.0 * correct) / (self.num_test)
        error = 100 - perc

        logging.info("[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)".format(
            correct, self.num_test, perc, error))
        return torch.cat(preds)

    def save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.

        If this model has reached the best validation accuracy thus
        far, a separate file with the suffix `best` is created.
        """
        filename = self.config.model_name + "_ckpt.pth.tar"
        ckpt_path = os.path.join(self.config.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        if is_best:
            filename = self.config.model_name + "_model_best.pth.tar"
            shutil.copyfile(ckpt_path, os.path.join(self.config.ckpt_dir, filename))

    def load_checkpoint(self, best=False):
        """Load the best copy of a model.
        Args:
            best: if set to True, loads the best model.
        """
        logging.info("[*] Loading model from {}".format(self.config.ckpt_dir))

        filename = self.config.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.config.model_name + "_model_best.pth.tar"
        ckpt_path = os.path.join(self.config.ckpt_dir, filename)
        logging.info(os.path.abspath(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # load variables from checkpoint
        self.start_epoch = ckpt["epoch"]
        self.best_valid_acc = ckpt["best_valid_acc"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])

        if best:
            logging.info(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt["epoch"], ckpt["best_valid_acc"]
                )
            )
        else:
            logging.info("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"]))
