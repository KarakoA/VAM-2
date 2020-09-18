import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def run():
    input = torch.tensor((1.0,2.0,3.0))
    log_softmax = F.log_softmax(input, dim = 0)
    softmax = F.softmax(input, dim = 0)

    print(log_softmax)
    print(softmax)
    print(log_softmax.argmax())
    print(softmax.argmax())

    a = torch.arange(12).reshape(3,4)
    print(a)
    print(torch.argmax(a,dim=1))

    R = torch.tensor((0,1,0)).float()
    R = R.unsqueeze(1).repeat(1, 4)
    print(R)

    y = torch.tensor((0)).unsqueeze(0)
    t1 = torch.tensor(((0.8,0.1,0.1))).unsqueeze(0)
    t2 = torch.tensor(((0.1, 0.8, 0.1))).unsqueeze(0)
    t3 = torch.tensor(((1.0, 0.0, 0.0))).unsqueeze(0)
    #t2 = torch.tensor((1,1,0))
    print(F.nll_loss(t1,y))
    print(F.nll_loss(t2, y))
    print(F.nll_loss(t3, y))

    means = torch.tensor((0.5,0.0,1.))
    sampled = torch.tensor((0.5,0.5,25.))
    sigma = 0.005
    r = torch.tensor((1,1,1))
    probs = Normal(means,sigma).log_prob(sampled)
    loss = torch.sum(-probs * r, dim = 0)
    print(probs)
    print(loss)
    #def loglikelihood(mean_arr, sampled_arr, sigma):
    #    mu = tf.stack(mean_arr)
    #    sampled = tf.stack(sampled_arr)
    #    gaussian = tf.contrib.distributions.Normal(mu, sigma)
    #    logll = gaussian.log_prob(sampled)
    #    logll = tf.reduce_sum(logll, 2)
    #    logll = tf.transpose(logll)
    #    return logll*/
    print(R)
    print(R.repeat(1, 2).reshape(3,4,2))
   #adjusted_reward.repeat(1, 2).reshape(self.config.batch_size, -1, 2).detach()
if __name__ == '__main__':
    run()