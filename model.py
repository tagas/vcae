import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from layers import _conv, _deconv, _linear

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr

base = importr('base')
rvinecop = importr('rvinecopulib')


class ae_vine(nn.Module):
    def __init__(self, label, image_size, channel_num, kernel_num, z_size, device):
        super().__init__()
        self.model_name = "ae_vine"
        self.label = label
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size
        self.device = device
        self.vine = None

        # encoder
        self.encoder = nn.Sequential(
            _conv(channel_num, kernel_num // 4),
            _conv(kernel_num // 4, kernel_num // 2),
            _conv(kernel_num // 2, kernel_num),
        )

        # encoded feature's size and volume
        self.feature_size = image_size // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # decoder
        self.decoder = nn.Sequential(
            _deconv(kernel_num, kernel_num // 2),
            _deconv(kernel_num // 2, kernel_num // 4),
            _deconv(kernel_num // 4, channel_num),
            nn.Sigmoid()
        )

        # projection
        self.project = _linear(z_size, self.feature_volume, relu=False)
        self.q_layer = _linear(self.feature_volume, z_size, relu=False)

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_layer(unrolled)

    def forward(self, x):
        encoded = self.encoder(x)
        # flatten and reshape for decoder
        z = self.q(encoded)

        x_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        x_reconstructed = self.decoder(x_projected)

        return x_reconstructed

    def sample(self, size, vine, noise=None):

            if noise is None:
                sampled_r = rvinecop.rvine(size, vine)
            else:
                sampled_r = rvinecop.inverse_rosenblatt(noise.cpu().numpy(), vine)

            sampled_py = torch.Tensor(np.asarray(sampled_r)).view(size, -1).to(self.device)

            # transform vine samples in shape for decoder
            sample_projected = self.project(sampled_py).view(
                -1, self.kernel_num,
                self.feature_size,
                self.feature_size,
            )
            del sampled_py, sampled_r
            output_vine = self.decoder(sample_projected)
            del sample_projected
            return output_vine

    @property
    def name(self):
            return (
                'ae_vine'
                '-{kernel_num}k'
                '-{label}'
                '-{channel_num}x{image_size}x{image_size}'
            ).format(
                label=self.label,
                kernel_num=self.kernel_num,
                image_size=self.image_size,
                channel_num=self.channel_num,
            )


class dec_vine(nn.Module):
    def __init__(self, label, image_size, channel_num, kernel_num, z_size,
                 cluster_number, device, alpha=1.0):
        super().__init__()
        self.ae = ae_vine(label,
                          image_size,
                          channel_num,
                          kernel_num,
                          z_size,
                          device)

        self.model_name = "dec_vine"
        self.label = label
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size
        self.vine = None
        self.cluster_number = cluster_number
        self.device = device
        self.alpha = alpha
        self.cluster_layer = Parameter(torch.Tensor(cluster_number, z_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, pretrain_path=''):
        if pretrain_path == '':
            raise ValueError('The pretrain_path argument for dec_vine is empty.')
        # load pretrain ae
        pretrained_ae = torch.load(pretrain_path, map_location=self.device)
        self.ae.load_state_dict(pretrained_ae['state'])
        print('load pretrained ae from', pretrain_path)

    def forward(self, x):
        # encoded representation
        z = self.ae.encoder(x)
        # flatten
        z = self.ae.q(z)
        # cluster
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # reshape for decoder
        output = self.ae.project(z).view(
            -1, self.kernel_num,
            self.ae.feature_size,
            self.ae.feature_size)
        # decode
        output = self.ae.decoder(output)
        return output, q

    def target_distribution(self, q):
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def sample(self, size, vine, noise=None):
        return self.ae.sample(size, vine, noise)

    @property
    def name(self):
            return (
                'dec_vine'
                '-{kernel_num}k'
                '-{label}'
                '-{channel_num}x{image_size}x{image_size}'
            ).format(
                label=self.label,
                kernel_num=self.kernel_num,
                image_size=self.image_size,
                channel_num=self.channel_num,
            )


class vae(nn.Module):
    def __init__(self, label, image_size, channel_num, kernel_num, z_size, device):
        # configurations
        super().__init__()
        self.model_name = "vae"
        self.label = label
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.device = device
        self.z_size = z_size

        # encoder
        self.encoder = nn.Sequential(
            _conv(channel_num, kernel_num // 4),
            _conv(kernel_num // 4, kernel_num // 2),
            _conv(kernel_num // 2, kernel_num),
        )

        # encoded feature's size and volume
        self.feature_size = image_size // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # q
        self.q_mean = _linear(self.feature_volume, z_size, relu=False)
        self.q_logvar = _linear(self.feature_volume, z_size, relu=False)

        # projection
        self.project = _linear(z_size, self.feature_volume, relu=False)

        # decoder
        self.decoder = nn.Sequential(
            _deconv(kernel_num, kernel_num // 2),
            _deconv(kernel_num // 2, kernel_num // 4),
            _deconv(kernel_num // 4, channel_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encode x
        encoded = self.encoder(x)

        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)
        z = self.z(mean, logvar)
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        # return the parameters of distribution of q given x and the
        # reconstructed image.
        return (mean, logvar), x_reconstructed

    # ==============
    # VAE components
    # ==============

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.randn(mean.shape)).to(self.device)
        return eps.mul(std).add_(mean)

    def kl_divergence_loss(self, mean, logvar):
        return ((mean**2 + logvar.exp() - 1 - logvar) / 2).mean()

    # =====
    # Utils
    # =====

    @property
    def name(self):
        return (
            'vae'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )

    def sample(self, size, noise=None):

        if noise is None:
            z = Variable(torch.randn(size, self.z_size)).to(self.device)
        else:
            z = noise

        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected).data




class ae_vine2(nn.Module):

    def __init__(self, image_size, hidden_dim, z_size, device, channel_num=3):
        super(ae_vine2, self).__init__()
        self.label = "ae_vine2"
        self.encoding_dim = z_size
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.model_name = "ae_vine2"
        self.vine = None
        self.z_size = z_size
        self.device = device
        self.channel_num = channel_num

        # Decoder - Fractional strided convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()  # nn.Tanh()
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

        # Fully-connected layers
        self.fc1 = nn.Linear(256, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.encoding_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.encoding_dim)
        self.fc3 = nn.Linear(self.encoding_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 256)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.sigmoid(self.fc4(h3))
        return self.decoder(h4.view(z.size(0), -1, 1, 1))

    def forward(self, x):
        # Encode
        encoded = F.relu(self.fc1(self.encoder(x).view(x.size(0), -1)))

        e = self.fc21(encoded)

        # Decode
        decoded = self.decode(e)

        return decoded

    def sample(self, size, vine, noise=None):

            if noise is None:
                sampled_r = rvinecop.rvine(size, vine)
            else:
                sampled_r = rvinecop.inverse_rosenblatt(noise.cpu().numpy(), vine)

            sampled_py = torch.Tensor(np.asarray(sampled_r)).view(size, -1).to(self.device)

            # Decode
            decoded = self.decode(sampled_py)

            del sampled_py

            return decoded

    @property
    def name(self):
        return (
            'ae_vine2'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )


class vae2(nn.Module):

    def __init__(self, image_size, hidden_dim, z_size, device, channel_num=3 ):
        super(vae2, self).__init__()

        self.label = "vae2"
        self.encoding_dim = z_size
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.model_name = "vae2"
        self.vine = None
        self.z_size = z_size
        self.device = device
        self.channel_num = channel_num

        # Decoder - Fractional strided convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()  # nn.Tanh()
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

        # Fully-connected layers
        self.fc1 = nn.Linear(256, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.encoding_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.encoding_dim)
        self.fc3 = nn.Linear(self.encoding_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 256)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.sigmoid(self.fc4(h3))
        return self.decoder(h4.view(z.size(0), -1, 1, 1))

    def forward(self, x):
        # Encode
        encoded = F.relu(self.fc1(self.encoder(x).view(x.size(0), -1)))

        # Obtain mu and logvar
        mu = self.fc21(encoded)
        logvar = self.fc22(encoded)

        # Reparametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        # Decode
        decoded = self.decode(z)

        # return decoded, mu, logvar
        return (mu, logvar), decoded

    def kl_divergence_loss(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    def sample(self, size, noise=None):
        if noise is None:
            z = Variable(torch.randn(size, self.z_size)).to(self.device)
        else:
            z = noise
        return self.decode(z)

    @property
    def name(self):
        return (
            'vae2'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )



class dec_vine2(nn.Module):
    def __init__(self, label, image_size, channel_num, kernel_num, z_size,
                 cluster_number, device, alpha=1.0):
        super().__init__()
        self.ae = ae_vine2(image_size=image_size,
                           hidden_dim=100,
                           z_size=z_size,
                           device=device,
                           channel_num=channel_num)

        self.model_name = "dec_vine2"
        self.label = label
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size
        self.vine = None
        self.cluster_number = cluster_number
        self.device = device
        self.alpha = alpha
        self.cluster_layer = Parameter(torch.Tensor(cluster_number, z_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, pretrain_path=''):
        if pretrain_path == '':
            raise ValueError('The pretrain_path argument for dec_vine is empty.')
        # load pretrain ae
        pretrained_ae = torch.load(pretrain_path, map_location=self.device)
        self.ae.load_state_dict(pretrained_ae['state'])
        print('load pretrained ae from', pretrain_path)

    def forward(self, x):

        # Encode
        z = F.relu(self.ae.fc1(self.ae.encoder(x).view(x.size(0), -1)))
        z = self.ae.fc21(z)

        # cluster
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # reshape for decoder

        # Decode
        output = self.ae.decode(z)

        return output, q

    def target_distribution(self, q):
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def sample(self, size, vine, noise=None):
        return self.ae.sample(size, vine, noise)

    @property
    def name(self):
            return (
                'dec_vine2'
                '-{kernel_num}k'
                '-{label}'
                '-{channel_num}x{image_size}x{image_size}'
            ).format(
                label=self.label,
                kernel_num=self.kernel_num,
                image_size=self.image_size,
                channel_num=self.channel_num,
            )


class ae_vine3(nn.Module):

    def __init__(self, image_size, hidden_dim, z_size, device, channel_num=3):
        super(ae_vine3, self).__init__()
        self.label = "ae_vine3"
        self.encoding_dim = z_size
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.model_name = "ae_vine3"
        self.vine = None
        self.z_size = z_size
        self.device = device
        self.channel_num = channel_num

        # Decoder - Fractional strided convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()  # nn.Tanh()
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

        # Fully-connected layers
        self.fc1 = nn.Linear(512, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.encoding_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.encoding_dim)
        self.fc3 = nn.Linear(self.encoding_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 512)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.sigmoid(self.fc4(h3))
        return self.decoder(h4.view(z.size(0), -1, 1, 1))

    def forward(self, x):
        # Encode
        encoded = F.relu(self.fc1(self.encoder(x).view(x.size(0), -1)))

        e = self.fc21(encoded)

        #print(e.shape)

        # Decode
        decoded = self.decode(e)
        #print(decoded.shape)

        return decoded

    def sample(self, size, vine, noise=None):

            if noise is None:
                sampled_r = rvinecop.rvine(size, vine, cores=4)
            else:
                sampled_r = rvinecop.inverse_rosenblatt(noise.cpu().numpy(), vine, cores=4)

            sampled_py = torch.Tensor(np.asarray(sampled_r)).view(size, -1).to(self.device)

            # Decode
            decoded = self.decode(sampled_py)

            del sampled_py

            return decoded

    @property
    def name(self):
        return (
            'ae_vine3'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )


class vae3(nn.Module):

    def __init__(self, image_size, hidden_dim, z_size, device, channel_num=3 ):
        super(vae3, self).__init__()

        self.label = "vae3"
        self.encoding_dim = z_size
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.model_name = "vae3"
        self.vine = None
        self.z_size = z_size
        self.device = device
        self.channel_num = channel_num

        # Decoder - Fractional strided convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()  # nn.Tanh()
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

        # Fully-connected layers
        self.fc1 = nn.Linear(512, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.encoding_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.encoding_dim)
        self.fc3 = nn.Linear(self.encoding_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 512)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.sigmoid(self.fc4(h3))
        return self.decoder(h4.view(z.size(0), -1, 1, 1))

    def forward(self, x):
        # Encode
        encoded = F.relu(self.fc1(self.encoder(x).view(x.size(0), -1)))

        # Obtain mu and logvar
        mu = self.fc21(encoded)
        logvar = self.fc22(encoded)

        # Reparametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        # Decode
        decoded = self.decode(z)

        # return decoded, mu, logvar
        return (mu, logvar), decoded

    def kl_divergence_loss(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    def sample(self, size, noise=None):
        if noise is None:
            z = Variable(torch.randn(size, self.z_size)).to(self.device)
        else:
            z = noise
        return self.decode(z)

    @property
    def name(self):
        return (
            'vae3'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )

class dec_vine3(nn.Module):
    def __init__(self, label, image_size, channel_num, kernel_num, z_size,
                 cluster_number, device, alpha=1.0):
        super().__init__()
        self.ae = ae_vine3(image_size=image_size,
                           hidden_dim=100,
                           z_size=z_size,
                           device=device,
                           channel_num=channel_num)

        self.model_name = "dec_vine3"
        self.label = label
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size
        self.vine = None
        self.cluster_number = cluster_number
        self.device = device
        self.alpha = alpha
        self.cluster_layer = Parameter(torch.Tensor(cluster_number, z_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, pretrain_path=''):
        if pretrain_path == '':
            raise ValueError('The pretrain_path argument for dec_vine is empty.')
        # load pretrain ae
        pretrained_ae = torch.load(pretrain_path, map_location=self.device)
        self.ae.load_state_dict(pretrained_ae['state'])
        print('load pretrained ae from', pretrain_path)

    def forward(self, x):

        # Encode
        z = F.relu(self.ae.fc1(self.ae.encoder(x).view(x.size(0), -1)))
        z = self.ae.fc21(z)

        # cluster
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # reshape for decoder

        # Decode
        output = self.ae.decode(z)

        return output, q

    def target_distribution(self, q):
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def sample(self, size, vine, noise=None):
        return self.ae.sample(size, vine, noise)

    @property
    def name(self):
            return (
                'dec_vine3'
                '-{kernel_num}k'
                '-{label}'
                '-{channel_num}x{image_size}x{image_size}'
            ).format(
                label=self.label,
                kernel_num=self.kernel_num,
                image_size=self.image_size,
                channel_num=self.channel_num,
            )


# Generator
class Generator(nn.Module):
    def __init__(self, latent=100, init_channel=32, img_channel=1):
        super().__init__()

        self.latent = latent
        self.init_channel = init_channel
        self.img_channel = img_channel

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.latent, self.init_channel * 8, 4, bias=False),
            nn.BatchNorm2d(self.init_channel * 8),
            nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(self.init_channel * 8, self.init_channel * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.init_channel * 4),
            nn.ReLU()
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(self.init_channel * 4, self.init_channel * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.init_channel * 2),
            nn.ReLU()
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(self.init_channel * 2, self.init_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.init_channel),
            nn.ReLU()
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(self.init_channel, self.img_channel, 4, 2, 1, bias=False),
            #nn.ConvTranspose2d(self.init_channel, self.img_channel, 1, 1, 0, bias=False),
            nn.Tanh()
        )

        # initialization for parameters

        for layer in self.modules():

            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.normal(layer.weight.data, 0, 0.02)

            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()
		# nn.init.normal(layer.weight.data, 0, 0.02)

    def forward(self, inputs):

        outputs = self.deconv1(inputs)
        outputs = self.deconv2(outputs)
        outputs = self.deconv3(outputs)
        outputs = self.deconv4(outputs)
        outputs = self.deconv5(outputs)

        return outputs


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, latent=100, init_channel=32, img_channel=1, slope=0.2):
        super().__init__()

        self.latent = latent
        self.init_channel = init_channel
        self.img_channel = img_channel
        self.slope = slope
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.img_channel, self.init_channel, 4, 2, 1, bias=False),
            nn.LeakyReLU(self.slope)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.init_channel, self.init_channel * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.init_channel * 2),
            nn.LeakyReLU(self.slope)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.init_channel * 2, self.init_channel * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.init_channel * 4),
            nn.LeakyReLU(self.slope)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.init_channel * 4, self.init_channel * 8, 4, 2, 1, bias=False),
            #nn.Conv2d(self.init_channel * 4, self.init_channel * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.init_channel * 8),
            nn.LeakyReLU(self.slope)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(self.init_channel * 8, 1, 4, 1, 0, bias=False), #celebA
            #nn.Conv2d(self.init_channel * 8, 1, 2, 2, 0, bias=False),            
            nn.Sigmoid()
        )

        # initialization for parameters
        for layer in self.modules():

            if isinstance(layer, nn.Conv2d):
                nn.init.normal(layer.weight.data, 0, 0.02)

            elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()
		# nn.init.normal(layer.weight.data, 1, 0.02)

    def forward(self, inputs):

        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)

        return outputs.view(inputs.size(0))



class gan(nn.Module):

    def __init__(self, latent=100, image_size=32, image_channel=1, init_channel=32):
        super().__init__()
        self.model_name = "gan"
        self.z_size = latent
        self.image_size = image_size
        self.channel_num = image_channel
        self.init_channel = init_channel

        self.net_g = Generator(self.z_size, self.init_channel, self.channel_num)
        self.net_d = Discriminator(self.z_size, self.init_channel, self.channel_num)

    def sample(self, noise=None):
        return 0.5 * self.net_g(noise).data.cpu() + 0.5
        #return self.net_g(noise).data.cpu() 


def get_noise(noise_num=64, latent=100):
     return Variable(torch.randn((noise_num, latent, 1, 1)))

