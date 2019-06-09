import os
import torch
import numpy as np
import utils
import torch.nn as nn
import torchvision
from tqdm import tqdm
from torch import optim
import metric as metric
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
from torchvision import datasets, transforms
from data import TRAIN_DATASETS, DATASET_CONFIGS, TEST_DATASETS
from sklearn.cluster import KMeans
from torch import distributions as dist
from model import get_noise


import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr

base = importr('base')
rvinecop = importr('rvinecopulib')


def train_model(model, dataset, ds_name,
                epochs=10,
                batch_size=32,
                sample_size=32,
                eval_size=32,
                img_size=32,
                lr=1e-3,
                weight_decay=1e-4,
                loss_log_interval=20,
                image_log_interval=20,
                model_log_interval=20,
                checkpoint_dir='./checkpoints',
                resume=False,
                cuda=False,
                seed=0,
                device=None,
                cores=1):
    if resume:
        epoch_start = utils.load_checkpoint(model, checkpoint_dir)
    else:
        epoch_start = 0

    fixed_noise = torch.rand(sample_size, model.z_size).to(device)

    if model.model_name in ['vae', 'cvae', 'vae2', 'cvae2']:
        m = dist.Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device))
        fixed_noise = m.icdf(fixed_noise)

    output_folder = './results/' + ds_name
    resfile_prefix = ds_name + "_" + \
                     model.model_name + \
                     "_ld_" + \
                     str(model.z_size) + \
                     "_bs_" + str(batch_size)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data_root = './datasets'

    if model.model_name in ['dec_vine', 'dec_vine2', 'dec_vine3']:

        # load pre-trained AE
        if model.model_name == 'dec_vine':
            pretrain_prefix = resfile_prefix.replace("dec", "ae")
        elif model.model_name == 'dec_vine2':
            pretrain_prefix = resfile_prefix.replace("dec_vine2", "ae_vine2")
        elif model.model_name == 'dec_vine3':
            pretrain_prefix = resfile_prefix.replace("dec_vine3", "ae_vine3")
        
        pretrain_files = [filename for filename in os.listdir(checkpoint_dir) if filename.startswith(pretrain_prefix)]
        pretrain_epochs = [int(filename.replace(pretrain_prefix + "_", "")) for filename in pretrain_files]
        pretrain_path = os.path.join(checkpoint_dir, pretrain_files[pretrain_epochs.index(max(pretrain_epochs))])
        model.pretrain(pretrain_path)

        # form initial cluster centres
        data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
        data_stream = tqdm(enumerate(data_loader, 1))
        features = []
        for batch_index, (x, _, _) in data_stream:

            tmp_x = Variable(x).to(device)
            if model.model_name == 'dec_vine':
                z = model.ae.encoder(tmp_x)
                z = model.ae.q(z)
            elif model.model_name == 'dec_vine2' or  model.model_name == 'dec_vine3':
                z = torch.nn.functional.relu(model.ae.fc1(model.ae.encoder(tmp_x).view(x.size(0), -1)))
                z = model.ae.fc21(z)

            features.append(z)

        kmeans = KMeans(n_clusters=model.cluster_number, n_init=20)
        y_pred = kmeans.fit_predict(torch.cat(features).detach().cpu().numpy())
        model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    pretrain=0
    if  pretrain==1 and model.model_name == 'ae_vine3':
        pretrain_prefix = resfile_prefix#ds_name + '_ae_vine3'
        pretrain_files = [filename for filename in os.listdir(checkpoint_dir) if filename.startswith(pretrain_prefix)]
        pretrain_epochs = [int(filename.replace(pretrain_prefix + "_", "")) for filename in pretrain_files]
        pretrain_path = os.path.join(checkpoint_dir, pretrain_files[pretrain_epochs.index(max(pretrain_epochs))])
        pretrained_ae = torch.load(pretrain_path, map_location=device)
        model.load_state_dict(pretrained_ae['state'])
        print('load pretrained ae3 from', pretrain_path)


    # reconstruction_criterion = torch.nn.BCELoss()
    reconstruction_criterion = torch.nn.BCELoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if model.model_name == 'gan':
        lr_g = lr_d = 0.0002
        k = 1
        fix_noise = get_noise(sample_size)
        opt_g = torch.optim.Adam(model.net_g.parameters(), lr=lr_g, betas=(0.5, 0.999))  # optimizer for Generator
        opt_d = torch.optim.Adam(model.net_d.parameters(), lr=lr_d, betas=(0.5, 0.999))  # optimizer for Discriminator

    for epoch in range(epoch_start, epochs + 1):
        print("Epoch {}".format(epoch))
        if model.model_name == "dec_vine" or model.model_name == "dec_vine2":
            # update target distribution p
            model.eval()
            p = []
            indices = []
            data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
            data_stream = tqdm(enumerate(data_loader, 1))

            for batch_index, (x, _, idx) in data_stream:
                tmp_x = Variable(x).to(device)
                _, tmp_p = model(tmp_x)
                p.append(tmp_p.detach().cpu())
                tmp_idx = idx
                indices.append(tmp_idx)

            p = torch.cat(p)
            indices = torch.cat(indices)
            p = model.target_distribution(p[indices])
            p = Variable(p).to(device)

        model.train()
        data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
        data_stream = tqdm(enumerate(data_loader, 1))

        for batch_index, (x, _, idx) in data_stream:
            
            # learning rate decay
            if  model.model_name == 'gan' and (epoch) == 8:# and dataset == "CelebA":
                    opt_g.param_groups[0]['lr'] /= 10
                    opt_d.param_groups[0]['lr'] /= 10
                    #print("learning rate change!")

            if model.model_name == 'gan' and (epoch) == 15: # and dataset == "CelebA":
                    opt_g.param_groups[0]['lr'] /= 10
                    opt_d.param_groups[0]['lr'] /= 10
                    #print("learning rate change!")            

            iteration = (epoch - 1) * (len(dataset) // batch_size) + batch_index
            x = Variable(x).to(device)
            idx = Variable(idx).to(device)


            if model.model_name == 'gan':
                # train Discriminator
                real_data = Variable(x.cuda())
                #print(real_data.shape)
                prob_fake = model.net_d(model.net_g(get_noise(real_data.size(0)).to(device)))
                prob_real = model.net_d(real_data)

                loss_d = - torch.mean(torch.log(prob_real) + torch.log(1 - prob_fake))

                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

                # train Generator
                if batch_index % k is 0:
                    prob_fake = model.net_d(model.net_g(get_noise().to(device)))

                    loss_g = - torch.mean(torch.log(prob_fake))

                    opt_g.zero_grad()
                    loss_g.backward()
                    opt_g.step()

            else:

                if model.model_name == 'ae_vine' or model.model_name == 'ae_vine2' or model.model_name == 'ae_vine3':
                    x_reconstructed = model(x)

                elif model.model_name == 'dec_vine' or model.model_name == 'dec_vine2':
                    x_reconstructed, q = model(x)
                    p_batch = p[idx]
                    penalization_loss = 10*F.kl_div(q.log(), p_batch)
                    del p_batch, q

                elif model.model_name == 'cvae' or model.model_name == "cvae2" or model.model_name=="cvae3":
                    (mean, logvar, atanhcor), x_reconstructed = model(x)
                    penalization_loss = model.kl_divergence_loss(mean, logvar, atanhcor)

                elif model.model_name == 'vae' or model.model_name == "vae2" or model.model_name=="vae3":
                    (mean, logvar), x_reconstructed = model(x)
                    penalization_loss = model.kl_divergence_loss(mean, logvar)

                reconstruction_loss = reconstruction_criterion(x_reconstructed, x) / x.size(0)

                if model.model_name == 'ae_vine' or model.model_name == 'ae_vine2' or model.model_name == 'ae_vine3':
                    loss = reconstruction_loss
                else:
                    loss = reconstruction_loss + penalization_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if iteration % loss_log_interval == 0:

                f = open(output_folder + "/" + resfile_prefix + "_losses" + ".txt", 'a')

                if model.model_name == 'gan':
                    f.write("\n{:<12} | {} | {} | {} | {} ".format(
                        model.model_name,
                        iteration,
                        loss_g,
                        loss_d,
                        seed
                    ))
                    '''                   
                    print("\n{:<12} | {} | {} | {} | {} ".format(
                        model.model_name,
                        iteration,
                        loss_g,
                        loss_d,
                        seed
                    ))
                    '''
                else:
                    if model.model_name == 'ae_vine' or model.model_name == 'ae_vine2' or model.model_name == 'ae_vine3':
                        f.write("\n{:<12} | {} | {} | {} ".format(
                            model.model_name,
                            iteration,
                            loss,
                            seed
                        ))

                    else:
                        f.write("\n{:<12} | {} | {} | {} | {} | {}".format(
                            model.model_name,
                            iteration,
                            reconstruction_loss.data.item(),
                            penalization_loss.data.item(),
                            loss.data.item(),
                            seed
                        ))

                f.close()

            # adding this just to have a way of calculating the scores at 0 epochs
            if batch_index > 0 and epoch == 0:
            	break

        if epoch % model_log_interval == 0:
            print()
            print('###################')
            print('# model checkpoint!')
            print('###################')
            print()
            utils.save_checkpoint(model, checkpoint_dir, epoch, resfile_prefix + "_" + str(epoch))

        if epoch % image_log_interval == 0:

            print()
            print('###################')
            print('# image checkpoint!')
            print('###################')
            print()

            model.eval()

            ae_vine_models = ['ae_vine', 'ae_vine2', 'dec_vine', 'dec_vine2', 'ae_vine3', 'dec_vine3']

            if model.model_name in ae_vine_models:

                data_loader_vine = utils.get_data_loader(dataset, 5000, cuda=cuda)
                data_stream_vine = tqdm(enumerate(data_loader_vine, 1))
                features = []

                for batch_index, (x, _, _) in data_stream_vine:

                    tmp_x = Variable(x).to(device)
                    if model.model_name == 'ae_vine':
                        encoded = model.encoder(tmp_x)
                        e = model.q(encoded)

                    elif model.model_name == 'dec_vine':
                        encoded = model.ae.encoder(tmp_x)
                        e = model.ae.q(encoded)

                    elif model.model_name == 'ae_vine2':
                        encoded = torch.nn.functional.relu(model.fc1(model.encoder(tmp_x).view(x.size(0), -1)))
                        e = model.fc21(encoded)

                    elif model.model_name == 'dec_vine2':
                        encoded = torch.nn.functional.relu(model.ae.fc1(model.ae.encoder(tmp_x).view(x.size(0), -1)))
                        e = model.ae.fc21(encoded)


                    elif model.model_name == 'ae_vine3':
                        encoded = F.relu(model.fc1(model.encoder(tmp_x).view(x.size(0), -1)))
                        e = model.fc21(encoded)

                    elif model.model_name == 'dec_vine3':
                        encoded = F.relu(model.ae.fc1(model.ae.encoder(tmp_x).view(x.size(0), -1)))
                        e = model.ae.fc21(encoded)
                    features.append(e.detach().cpu())
                    if batch_index > 0:
                        break

                features = torch.cat(features).numpy()
                #np.savetxt(resfile_prefix + '_features' + str(epoch) + '_.csv', features, delimiter=",")
                copula_controls = base.list(family_set="tll", trunc_lvl=5, cores=cores)
                vine_obj = rvinecop.vine(features, copula_controls=copula_controls)

                model.vine = vine_obj

                fake = model.sample(sample_size, vine_obj, fixed_noise)

                del x, e, encoded, vine_obj,data_loader_vine

            elif model.model_name == 'gan':
                fake = model.net_g(fix_noise.to(device)).data.cpu() #+ 0.5
                print(fake.shape)
            else:

                fake = model.sample(sample_size, fixed_noise)

            fake = fake.reshape(sample_size, model.channel_num,
                                model.image_size, model.image_size)
            name_str = resfile_prefix + '_fake_samples_epoch'
            vutils.save_image(fake.detach(),
                              '%s/%s_%03d.png' % (output_folder, name_str, epoch),
                              normalize=True)
            del fake
        
        if epoch % 10 == 0: 
            
            s = metric.compute_score_raw(ds_name, dataset, img_size, data_root,
                                         eval_size, batch_size,
                                         output_folder + '/real/',
                                         output_folder + '/fake/',
                                         model, model.z_size, 'resnet34', device)

            f = open(output_folder + "/" + resfile_prefix + "_scores" + ".txt", 'a')

            scr_arr = [str(a) for a in s]
            f.write("\n{:<12} | {} | {} | {}".format(
                model.model_name,
                epoch,
                ', '.join(scr_arr),
                seed
            ))

            f.close()
            
         
