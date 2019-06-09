Contents of this folder are as follows:
.
├── README (this file)
├── cvine_sample.py
├── data.py (downloading and data transformations)
├── layers.py (convolutional layers for AE)
├── main.py (starting point for the experiments)
├── metric.py (functions for calcucating multiple scores, based on https://arxiv.org/pdf/1806.07755.pdf)
├── model.py (stores all generative models ae_vine, vae, cvae, dec_vine and their dcgan versions, as dcgan itself)
├── train.py (most of the logics/execution for the experiment is done here)
├── utils.py (loading datasets, saving models)
└── utils_r.R (functions used only in the R version of cvae/cvae2)


A simple demo for VCAE is available by just running the main.py script.
Note that this code requires Python 3.6, pytorch 4.0 and R 3.6 installed on your machine.