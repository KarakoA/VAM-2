import unittest

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal


class MyTestCase(unittest.TestCase):

    def reinfoce_loss_f(self,means, locs, rewards):
        probs = Normal(means, 0.05).log_prob(locs)
        print(probs)
        # summed over timesteps and averaged across batch
        # - log probs, because loss and not reward
        loss_reinforce = - torch.sum(probs * rewards, dim=1)#.sum(dim=0)
        #Assmume no batch
        #loss_reinforce = torch.mean(loss_reinforce, dim=0)
        return loss_reinforce

    # fixed reward:
    # 0.0 see what happens if means == locs or if means != locs

    def test_something(self):
        rewards = torch.tensor(((0.0, 0.0, 0.0),
                                (1.0, 1.0, 1.0),
                                (0.0, 0.0, 0.0),
                                (0.0, 0.0, 0.0)))

        means = torch.tensor(((0.0, 0.0, 0.0),
                              (1.0, 1.0, 1.0),
                              (0.0, 0.0, 0.0),
                              (0.0, 0.0, 0.0)))

        locs = torch.tensor(((0.0, 0.0, 0.0),
                             (1.0, 1.0, 1.0),
                             (0.0, 0.0, 0.0),
                             (0.0, 0.0, 0.0)))
        r1= self.reinfoce_loss_f(means,locs,rewards)
        print(r1)
        self.assertEqual(True, False)

    def test_show_some_probs(self):
        p1 = Normal(0.5, 0.05).log_prob(0.5)
        p2 = Normal(0.5, 0.05).log_prob(1)
        p3 = Normal(0.5, 0.05).log_prob(0)
        print(p1)
        print(p2)
        print(p3)
        self.assertGreater(p1,p3)
        self.assertGreater(p1,p2)


# in reality never happens, are close
    def test_something_2(self):
        rewards = torch.tensor(((0.0, 0.0, 0.0),
                                (1.0, 1.0, 1.0),
                                (0.0, 0.0, 0.0),
                                (0.0, 0.0, 0.0)))

        means = torch.tensor(((0.0, 0.0, 0.0),
                              (1.0, 1.0, 1.0),
                              (0.0, 0.0, 0.0),
                              (0.0, 0.0, 0.0)))

        locs = torch.tensor(((0.0, 0.0, 0.0),
                             (1.0, 1.0, 200.0),
                             (0.0, 0.0, 0.0),
                             (0.0, 0.0, 0.0)))
        r1= self.reinfoce_loss_f(means,locs,rewards)
        print(r1)
        self.assertEqual(True, False)

    def test_something_3(self):
        rewards = torch.tensor(((0.0, 0.0, 0.0),
                                (0.0, 0.0, 0.0),
                                (0.0, 0.0, 0.0),
                                (0.0, 0.0, 0.0)))

        means = torch.tensor(((0.0, 0.0, 0.0),
                              (1.0, 1.0, 1.0),
                              (0.0, 0.0, 0.0),
                              (0.0, 0.0, 0.0)))

        locs = torch.tensor(((0.0, 0.0, 0.0),
                             (1.0, 1.0, 200.0),
                             (0.0, 0.0, 0.0),
                             (0.0, 0.0, 0.0)))
        r1= self.reinfoce_loss_f(means,locs,rewards)
        print(r1)
        self.assertEqual(True, False)


    def test_something_4(self):
        rewards = torch.tensor(((0.2, 0.2, 0.2),
                                (1.0, 1.0, 1.0),
                                (0.1, 0.1, 0.1),
                                (0.0, 0.0, 0.0)))

        means = torch.tensor(((0.0, 0.0, 0.0),
                              (1.0, 1.0, 1.0),
                              (0.0, 0.0, 0.0),
                              (0.0, 0.0, 0.0)))

        locs = torch.tensor(((0.0, 0.0, 0.0),
                             (1.0, 1.0, 1.0),
                             (0.0, 0.0, 0.0),
                             (0.0, 0.0, 0.0)))
        r1 = self.reinfoce_loss_f(means, locs, rewards)
        print(r1)
        self.assertEqual(True, False)

    # 2 should have the smallest loss (because greatest reward), so should be forged in direction of 2
    def test_something_5(self):
        rewards = torch.tensor(((0.2, 0.2, 0.2),
                                (1.0, 1.0, 1.0),
                                (0.1, 0.1, 0.1),
                                (0.0, 0.0, 0.0)))

        means = torch.tensor(((0.0, 0.0, 0.0),
                              (1.0, 1.0, 1.0),
                              (0.0, 0.0, 0.0),
                              (0.0, 0.0, 0.0)))

        locs = torch.tensor(((0.0, 0.0, 0.0),
                             (1.0, 1.0, 1.0),
                             (0.0, 0.0, 0.0),
                             (0.0, 0.0, 0.0)))
        r1 = self.reinfoce_loss_f(means, locs, rewards)
        print(r1)


        self.assertEqual(True, False)

if __name__ == '__main__':
    unittest.main()
