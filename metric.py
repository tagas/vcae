import os
import ot
import pdb
import math
import timeit

from scipy.stats import entropy
from numpy.linalg import norm
from scipy import linalg
import numpy as np

import torch
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms

from data import TRAIN_DATASETS,TEST_DATASETS, DATASET_CONFIGS
from model import get_noise

def give_name(iter):  # 7 digit name.
    ans = str(iter)
    return ans.zfill(7)


def sample_fake(model, nz, sample_size, batch_size, save_folder, device="cpu"):
    print('sampling fake images ...')
    save_folder = save_folder + '/0/'
    try:
        os.makedirs(save_folder)
    except OSError:
        pass

    ae_vine_models = ['ae_vine', 'ae_vine2', 'dec_vine', 'dec_vine2','ae_vine3', 'dec_vine3']

    if model.model_name == 'gan':

        iter = 0
        for i in range(0, 1 + sample_size):
            noise = get_noise(1).to(device)
            fake = 0.5*model.sample(noise) + 0.5
            fake = fake.reshape(1, model.channel_num,
                                model.image_size, model.image_size)

            for j in range(0, len(fake.data)):
                if iter < sample_size:
                    vutils.save_image(fake.data[j],
                                      save_folder + give_name(iter) + ".png",
                                      normalize=True)
                iter += 1
                if iter >= sample_size:
                    break

            del fake

    else:
        if model.model_name in ae_vine_models:
            fake = model.sample(sample_size, model.vine)
            fake = fake.reshape(sample_size, model.channel_num,
                                model.image_size, model.image_size)
            iter = 0
            for j in range(0, len(fake.data)):
                vutils.save_image(fake.data[j],
                                  save_folder + give_name(iter) + ".png",
                                  normalize=True)
                iter += 1

        else:
            noise = torch.FloatTensor(batch_size, nz, 1, 1).to(device)
            iter = 0
            for i in range(0, 1 + sample_size // batch_size):
                noise.data.normal_(0, 1)

                fake = model.sample(sample_size)
                fake = fake.reshape(sample_size, model.channel_num,
                                    model.image_size, model.image_size)

                for j in range(0, len(fake.data)):
                    if iter < sample_size:
                        vutils.save_image(fake.data[j],
                                          save_folder + give_name(iter) + ".png",
                                          normalize=True)
                    iter += 1
                    if iter >= sample_size:
                        break

                del fake

def sample_true(dataset, image_size, data_root, sample_size, batch_size, save_folder):
    print('sampling real images ...')
    save_folder = save_folder + '/0/'
    workers = 4
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                             batch_size=batch_size,
                                             num_workers=int(workers))

    if not os.path.exists(save_folder):
        try:
            os.makedirs(save_folder)
        except OSError:
            pass

        iter = 0
        for i, data in enumerate(dataloader, 0):
            img, _ = data
            for j in range(0, len(img)):
                vutils.save_image(img[j], save_folder + give_name(iter) + ".png")
                iter += 1
                if iter >= sample_size:
                    break
            if iter >= sample_size:
                break

def sample_true_test(ds_name, image_size, data_root, sample_size, batch_size, save_folder):
    print('sampling real images ...')
    save_folder = save_folder + '/0/'
    workers = 4
    # assert dataset
    dataset = TEST_DATASETS[ds_name]

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                             batch_size=sample_size,
                                             num_workers=int(workers))
  
    if not os.path.exists(save_folder):
        try:
            os.makedirs(save_folder)
        except OSError:
            pass

        iter = 0
        for i, data in enumerate(dataloader, 0):
            img, _, _ = data
            for j in range(0, len(img)):
                vutils.save_image(img[j], save_folder + give_name(iter) + ".png")
                iter += 1
                if iter >= sample_size:
                    break
            if iter >= sample_size:
                break

class ConvNetFeatureSaver(object):
    def __init__(self, device, model='resnet34', workers=4, batch_size=64):
        '''
        model: inception_v3, vgg13, vgg16, vgg19, resnet18, resnet34,
               resnet50, resnet101, or resnet152
        '''
        self.model = model
        self.batch_size = batch_size
        self.workers = workers
        if self.model.find('vgg') >= 0:
            self.vgg = getattr(models, model)(pretrained=True).to(device).eval()
            self.trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model.find('resnet') >= 0:
            resnet = getattr(models, model)(pretrained=True)
            resnet.to(device).eval()
            resnet_feature = nn.Sequential(resnet.conv1, resnet.bn1,
                                           resnet.relu,
                                           resnet.maxpool, resnet.layer1,
                                           resnet.layer2, resnet.layer3,
                                           resnet.layer4).to(device).eval()
            self.resnet = resnet
            self.resnet_feature = resnet_feature
            self.trans = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.model == 'inception' or self.model == 'inception_v3':
            inception = models.inception_v3(
                pretrained=True, transform_input=False).to(device).eval()
            inception_feature = nn.Sequential(inception.Conv2d_1a_3x3,
                                              inception.Conv2d_2a_3x3,
                                              inception.Conv2d_2b_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Conv2d_3b_1x1,
                                              inception.Conv2d_4a_3x3,
                                              nn.MaxPool2d(3, 2),
                                              inception.Mixed_5b,
                                              inception.Mixed_5c,
                                              inception.Mixed_5d,
                                              inception.Mixed_6a,
                                              inception.Mixed_6b,
                                              inception.Mixed_6c,
                                              inception.Mixed_6d,
                                              inception.Mixed_7a,
                                              inception.Mixed_7b,
                                              inception.Mixed_7c,
                                              ).to(device).eval()
            self.inception = inception
            self.inception_feature = inception_feature
            self.trans = transforms.Compose([
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            raise NotImplementedError

    def extract(self, device, imgFolder, save2disk=False):
        dataset = dset.ImageFolder(root=imgFolder, transform=self.trans)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.workers)
        print('extracting features...')
        feature_pixl, feature_conv, feature_smax, feature_logit = [], [], [], []
        for img, _ in tqdm(dataloader):
            with torch.no_grad():
                input = img.to(device)
                if self.model == 'vgg' or self.model == 'vgg16':
                    fconv = self.vgg.features(input).view(input.size(0), -1)
                    flogit = self.vgg.classifier(fconv)
                    # flogit = self.vgg.logitifier(fconv)
                elif self.model.find('resnet') >= 0:
                    fconv = self.resnet_feature(
                        input).mean(3).mean(2).squeeze()
                    flogit = self.resnet.fc(fconv)
                elif self.model == 'inception' or self.model == 'inception_v3':
                    fconv = self.inception_feature(
                        input).mean(3).mean(2).squeeze()
                    flogit = self.inception.fc(fconv)
                else:
                    raise NotImplementedError
                fsmax = F.softmax(flogit)
                feature_pixl.append(img)
                feature_conv.append(fconv.data.cpu())
                feature_logit.append(flogit.data.cpu())
                feature_smax.append(fsmax.data.cpu())

        feature_pixl = torch.cat(feature_pixl, 0)
        feature_conv = torch.cat(feature_conv, 0)
        feature_logit = torch.cat(feature_logit, 0)
        feature_smax = torch.cat(feature_smax, 0)

        if save2disk:
            torch.save(feature_pixl, os.path.join(
                imgFolder, 'feature_pixl.pth'))
            torch.save(feature_conv, os.path.join(
                imgFolder, 'feature_conv.pth'))
            torch.save(feature_logit, os.path.join(
                imgFolder, 'feature_logit.pth'))
            torch.save(feature_smax, os.path.join(
                imgFolder, 'feature_smax.pth'))

        return feature_pixl, feature_conv, feature_logit, feature_smax


def distance(X, Y, sqrt, device):
    device = "cpu"
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX,-1).to(device)
    X2 = (X*X).sum(1).resize_(nX,1)
    Y = Y.view(nY,-1).to(device)
    Y2 = (Y*Y).sum(1).resize_(nY,1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
            2 * torch.mm(X, Y.transpose(0, 1)))

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M


def wasserstein(M, sqrt):
    if sqrt:
        M = M.abs().sqrt()
    emd = ot.emd2([], [], M.numpy())

    return emd


class Score_knn:
    acc = 0
    acc_real = 0
    acc_fake = 0
    precision = 0
    recall = 0
    tp = 0
    fp = 0
    fn = 0
    ft = 0


def knn(Mxx, Mxy, Myy, k, sqrt):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1)))
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat(
        (Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1))
                ).topk(k, 0, False)

    count = torch.zeros(n0 + n1)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1)).float()

    s = Score_knn()
    s.tp = (pred * label).sum()
    s.fp = (pred * (1 - label)).sum()
    s.fn = ((1 - pred) * label).sum()
    s.tn = ((1 - pred) * (1 - label)).sum()
    s.precision = s.tp / (s.tp + s.fp + 1e-10)
    s.recall = s.tp / (s.tp + s.fn + 1e-10)
    s.acc_t = s.tp / (s.tp + s.fn)
    s.acc_f = s.tn / (s.tn + s.fp)
    s.acc = torch.eq(label, pred).float().mean()
    s.k = k

    return s


def mmd(Mxx, Mxy, Myy, sigma):
    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx / (scale * 2 * sigma * sigma))
    Mxy = torch.exp(-Mxy / (scale * 2 * sigma * sigma))
    Myy = torch.exp(-Myy / (scale * 2 * sigma * sigma))
    mmd = math.sqrt(Mxx.mean() + Myy.mean() - 2 * Mxy.mean())

    return mmd


def entropy_score(X, Y, epsilons, device):
    Mxy = distance(X, Y, False, device)
    scores = []
    for epsilon in epsilons:
        scores.append(ent(Mxy.t(), epsilon))

    return scores


eps = 1e-20
def inception_score(X):
    kl = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    score = np.exp(kl.sum(1).mean())

    return score

def mode_score(X, Y):
    kl1 = X * ((X+eps).log()-(X.mean(0)+eps).log().expand_as(X))
    kl2 = X.mean(0) * ((X.mean(0)+eps).log()-(Y.mean(0)+eps).log())
    score = np.exp(kl1.sum(1).mean() - kl2.sum())

    return score


def fid(X, Y):
    m = X.mean(0)
    m_w = Y.mean(0)
    X_np = X.numpy()
    Y_np = Y.numpy()

    C = np.cov(X_np.transpose())
    C_w = np.cov(Y_np.transpose())
    C_C_w_sqrt = linalg.sqrtm(C.dot(C_w), True).real

    score = m.dot(m) + m_w.dot(m_w) - 2 * m_w.dot(m) + \
        np.trace(C + C_w - 2 * C_C_w_sqrt)
    return np.sqrt(score)


class Score:
    emd = 0
    mmd = 0
    knn = None


def compute_score(real, fake, device, k=1, sigma=1, sqrt=True):

    Mxx = distance(real, real, False, device)
    Mxy = distance(real, fake, False, device)
    Myy = distance(fake, fake, False, device)

    s = Score()
    s.emd = wasserstein(Mxy, sqrt)
    s.mmd = mmd(Mxx, Mxy, Myy, sigma)
    s.knn = knn(Mxx, Mxy, Myy, k, sqrt)

    return s


def compute_score_raw(ds_name, dataset, image_size, data_root, sample_size, batch_size,
                      save_folder_r, save_folder_f, netG, nz,
                      conv_model='resnet34',
                      device="cpu"):

    sample_true_test(ds_name, image_size, data_root, sample_size, batch_size,
               save_folder_r)
    sample_fake(netG, nz, sample_size, batch_size, save_folder_f, device)

    convnet_feature_saver = ConvNetFeatureSaver(device,
                                                model=conv_model,
                                                batch_size=batch_size)
    feature_r = convnet_feature_saver.extract(device, save_folder_r)
    feature_f = convnet_feature_saver.extract(device, save_folder_f)

    # 4 feature spaces and 7 scores + incep + modescore + fid
    score = np.zeros(4 * 7 + 3)
    for i in range(0, 4):
        print('compute score in space: ' + str(i))
        Mxx = distance(feature_r[i], feature_r[i], False, device)
        Mxy = distance(feature_r[i], feature_f[i], False, device)
        Myy = distance(feature_f[i], feature_f[i], False, device)

        score[i * 7] = wasserstein(Mxy, True)
        score[i * 7 + 1] = mmd(Mxx, Mxy, Myy, 1)
        tmp = knn(Mxx, Mxy, Myy, 1, False)
        score[(i * 7 + 2):(i * 7 + 7)] = \
            tmp.acc, tmp.acc_t, tmp.acc_f, tmp.precision, tmp.recall

    score[28] = inception_score(feature_f[3])
    score[29] = mode_score(feature_r[3], feature_f[3])
    score[30] = fid(feature_r[3], feature_f[3])
    return score


def compute_score_short(ds_name, dataset, image_size, data_root, sample_size, batch_size,
                      save_folder_r, save_folder_f, netG, nz,
                      conv_model='resnet34',
                      device="cpu"):

    """
    Calculate only Inception score, Mode score and FID
    """

    sample_true_test(ds_name, image_size, data_root, sample_size, batch_size,
                     save_folder_r)
    sample_fake(netG, nz, sample_size, batch_size, save_folder_f, device)

    convnet_feature_saver = ConvNetFeatureSaver(device,
                                                model=conv_model,
                                                batch_size=batch_size)
    feature_r = convnet_feature_saver.extract(device, save_folder_r)
    feature_f = convnet_feature_saver.extract(device, save_folder_f)

    score = np.zeros(3)

    score[0] = inception_score(feature_f[3])
    score[1] = mode_score(feature_r[3], feature_f[3])
    score[2] = fid(feature_r[3], feature_f[3])
    return score

