import torch
import torch.nn.functional as nn
import torch.nn as nn2
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64
Z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return torch.randn(*size) * xavier_stddev

# =============================== Q(z|X) ======================================

'''
Wxh = xavier_init(size=[X_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True).cuda()

Whz_mu = xavier_init(size=[h_dim, Z_dim])
bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True).cuda()

Whz_var = xavier_init(size=[h_dim, Z_dim])
bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True).cuda()


def Q(X):a
    h = nn.relu(X @ Wxh + bxh.repeat(X.size(0), 1))
    z_mu = h @ Whz_mu + bhz_mu.repeat(h.size(0), 1)
    z_var = h @ Whz_var + bhz_var.repeat(h.size(0), 1)
    return z_mu, z_var


def sample_z(mu, log_var):
    eps = Variable(torch.randn(mb_size, Z_dim))
    return mu + torch.exp(log_var / 2) * eps
'''

# =============================== P(X|z) ======================================

'''
Wzh = xavier_init(size=[Z_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True).cuda()

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True).cuda()


def P(z):
    h = nn.relu(z @ Wzh + bzh.repeat(z.size(0), 1))
    X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    return X

'''
def sample_z(mu, log_var):
		eps = Variable(torch.randn(mb_size, Z_dim))
		return mu + torch.exp(log_var / 2) * eps

class SimpleNN(torch.nn.Module):
	def __init__(self,X_dim,h_dim,Z_dim):
		super(SimpleNN,self).__init__()
		self.Wxh=nn2.Parameter(xavier_init(size=[X_dim, h_dim]))
		self.bxh=nn2.Parameter(torch.zeros(h_dim))
		self.Whz_mu = nn2.Parameter(xavier_init(size=[h_dim, Z_dim]))
		self.bhz_mu = nn2.Parameter(torch.zeros(Z_dim))
		self.Whz_var = nn2.Parameter(xavier_init(size=[h_dim, Z_dim]))
		self.bhz_var = nn2.Parameter(torch.zeros(Z_dim))
		self.Wzh = nn2.Parameter(xavier_init(size=[Z_dim, h_dim]))
		self.bzh = nn2.Parameter(torch.zeros(h_dim))
		self.Whx = nn2.Parameter(xavier_init(size=[h_dim, X_dim]))
		self.bhx = nn2.Parameter(torch.zeros(X_dim))
	
	def Q(self,X):
		h = nn.relu(X @ self.Wxh + self.bxh.repeat(X.size(0), 1))
		z_mu = h @ self.Whz_mu + self.bhz_mu.repeat(h.size(0), 1)
		z_var = h @ self.Whz_var + self.bhz_var.repeat(h.size(0), 1)
		return z_mu, z_var
	
	def P(self,z):
		h = nn.relu(z @ self.Wzh + self.bzh.repeat(z.size(0), 1))
		X = nn.sigmoid(h @ self.Whx + self.bhx.repeat(h.size(0), 1))
		return X
	
	def forward(self, X):
		z_mu, z_var = self.Q(X)
		#z_mu=Variable(z_mu)
		#z_var=Variable(z_var)
		eps = Variable(torch.randn(mb_size, Z_dim)).cuda()
		z = z_mu + torch.exp(z_var / 2) * eps
		#z = Variable(sample_z(z_mu, z_var)).cuda()
		X_sample = self.P(z)
		return z_mu,z_var,z,X_sample

model=SimpleNN(X_dim,h_dim,Z_dim).cuda()

# =============================== TRAINING ====================================

#params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var,
#          Wzh, bzh, Whx, bhx]

solver = optim.Adam(model.parameters(),lr=lr)

for it in range(5000):
    X, _ = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X)).cuda()
    solver.zero_grad()
    # Forward
    #z_mu, z_var = Q(X)
    #z = sample_z(z_mu, z_var)
    #X_sample = P(z)

    z_mu,z_var,z,X_sample=model(X)
    # Loss
    recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False) / mb_size
    kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
    loss = (recon_loss + kl_loss).cuda()

    # Backward
    loss.backward()

    # Update
    solver.step()

    # Housekeeping
    #for p in params:
    #    if p.grad is not None:
    #        data = p.grad.data
    #        p.grad = Variable(data.new().resize_as_(data).zero_())

    #[o.zero_grad() for o in model.parameters()]
    
    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; Loss: {:.4}'.format(it, loss.data[0]))
        z = z.cuda()
        samples = model.P(z).cpu().data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
        c += 1
        plt.close(fig)
