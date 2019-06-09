#!/usr/bin/env python3

import os
import argparse
import torch
import torchvision
from train import train_model
from utils import reset_seeds
from data import TRAIN_DATASETS, DATASET_CONFIGS
from model import ae_vine, ae_vine2, dec_vine, dec_vine2, vae, vae2,vae3, cvae, gan#,# cvae2,cvae3,cvae3,cvae3, ae_vine3, dec_vine3, gan_2


parser = argparse.ArgumentParser(description='Experiment 1')

parser.add_argument('--dataset', default='mnist',
                    choices=list(TRAIN_DATASETS.keys()))
parser.add_argument('--model', type=str, default='ae_vine')
parser.add_argument('--kernel-num', type=int, default=128)
parser.add_argument('--z-size', type=int, default=10)
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--sample-size', type=int, default=64)
parser.add_argument('--image-size', type=int, default=32)
parser.add_argument('--evaluation_size', type=int, default=2000)

parser.add_argument('--lr', type=float, default=5e-03)
parser.add_argument('--weight-decay', type=float, default=1e-03)

parser.add_argument('--loss-log-interval', type=int, default=100)
parser.add_argument('--image-log-interval', type=int, default=20)
parser.add_argument('--model-log-interval', type=int, default=50)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
parser.add_argument('--results-dir', type=str, default='./results')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')
parser.add_argument('--vine-cpu-cores', type=int, default=4)

if __name__ == '__main__':
    args = parser.parse_args()
    seed = args.seed
    reset_seeds(seed)
    model_name = globals()[args.model]
    cuda = args.cuda and torch.cuda.is_available()
    dataset_config = DATASET_CONFIGS[args.dataset]
    dataset = TRAIN_DATASETS[args.dataset]
    image_size = args.image_size
    evaluation_sample_size = args.evaluation_size

    device = "cpu"

    if cuda:
        print("### CUDA is available ###")
        device = torch.device("cuda")

    else:
        print("### CUDA is NOT available ###")
        device = torch.device("cpu")

    if args.dataset in ['svhn', 'mnist']:
        gan_type = 0
        init_channel = 16
    else:
        gan_type = 1
        init_channel = 64#128#128

    print("Gan type ", gan_type)
    if args.model in ["ae_vine", "vae", "cvae"]:
        model = model_name(
            label=args.dataset,
            image_size=dataset_config['size'],
            channel_num=dataset_config['channels'],
            kernel_num=args.kernel_num,
            z_size=args.z_size,
            device=device
        )

    elif args.model in ["dec_vine", "dec_vine2", "dec_vine3"]:
        model = model_name(
            label=args.dataset,
            image_size=dataset_config['size'],
            channel_num=dataset_config['channels'],
            kernel_num=args.kernel_num,
            z_size=args.z_size,
            cluster_number=dataset_config['classes'],
            device=device
        )

    elif args.model in ["ae_vine2", "vae2", "vae3",  "cvae2", "cvae3", "ae_vine2c", "ae_vine3"]:
        model = model_name(
            image_size=dataset_config['size'],
            hidden_dim=100,
            z_size=args.z_size,
            device=device,
            channel_num=dataset_config['channels'])

    elif args.model in ['gan']:
        model = model_name(latent=args.z_size,
                           image_size=dataset_config['size'],
                           image_channel=dataset_config['channels'],
                           init_channel=init_channel,
			   gan_type=gan_type)

    model.to(device)
    print(model)

    print()
    print("Model name: ", model.model_name)
    print()
    print("Latent dimension = ", args.z_size)
    print()

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    train_model(
            model, dataset=dataset,
            ds_name=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            eval_size=evaluation_sample_size,
            img_size=image_size,
            lr=1e-3,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir,
            loss_log_interval=args.loss_log_interval,
            image_log_interval=args.image_log_interval,
            model_log_interval=args.model_log_interval,
            resume=args.resume,
            cuda=cuda,
            seed=seed,
            device=device,
            cores=args.vine_cpu_cores
    )

