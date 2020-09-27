import numpy as np
import torch
from scipy.stats import norm

from torch.distributions import Normal

from utils.graph_viz import make_dot


def main():
    mean = 1.0
    sigma = 0.25

    #input = np.array((0.5,0.75,1,1.25,1.5))
    x = np.array((0.5))
    probs = norm.pdf(x,loc = mean, scale = sigma)
    # natural log
    u = probs
    #print(np.log(probs))
    print(norm.logpdf(x,loc = mean ,scale = sigma ))
    mean  = torch.tensor(mean, requires_grad=True)
    probs = Normal(mean, torch.tensor(sigma)).log_prob(torch.tensor(x))
    print(probs)
    #print(sum(probs))
    probs.backward()
    print(mean.grad)

    #dot = make_dot(probs)
    #dot.render('./tmp/dot-graph-3.gv', view = True)
    print((x - u/(sigma* sigma)))
    # missing * e

    # x-u / s^2

    # d ln(x) = 1/x

    # d f(x


#F = -((value - self.loc) ** 2) / (2 * var) - math.log(self.scale) - math.log(math.sqrt(2 * math.pi))
# dF/self.loc  =
# -((value - self.loc) ** 2) / (2 * var)
#

    print((0.5 - 1.0) / 0.25 * 0.25)

def main2():
    mean = 1.0
    sigma = 0.25
    x = 0.5
    mean_torch  = torch.tensor(mean, requires_grad=True)
    probs = Normal(mean_torch, torch.tensor(sigma)).log_prob(torch.tensor(x))
    print(f"ln(pdf(x)) numpy: {norm.logpdf(x, loc=mean, scale=sigma)}, pytorch: {probs}")
    probs.backward()


    grad_numpy = (x - mean)/ (sigma * sigma)
    print(f"Gradient w.r.t mean numpy: {grad_numpy} pytorch: {mean_torch.grad}")
if __name__ == '__main__':
    main2()