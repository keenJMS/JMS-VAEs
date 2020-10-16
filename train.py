import os 
import torch.nn
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from utils.utils import *
from IPython import embed
from models import *
import argparse
#config
config=get_config('/root/proj/JMS-VAEs/config/base_setting.yml')
parser = argparse.ArgumentParser(description='Train image model ')
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed',default=1002)
args = parser.parse_args()
def main():


    #gpu_setting
    use_gpu=torch.cuda.is_available()
    pin_memory=True if use_gpu else False
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    dataset=torchvision.datasets.MNIST(root='root/dataset',
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)

    data_loader=torch.utils.data.DataLoader(dataset=dataset,
                                        batch_size=config['Trainer']['batch_size'],
                                        shuffle=True)
    model=VAE()
    opt=torch.optim.Adam(model.parameters(),lr=config['Trainer']['lr'])
    if use_gpu:
        model=model.cuda()
    for epoch in range(config['Trainer']['max_epoch']):
        train(model,opt,data_loader,config['Trainer']['max_epoch'],use_gpu,epoch)
def train(model,opt,data_loader,max_epoch,use_gpu,epoch):
    model.train()
    for batch_id,(imgs,labels) in enumerate(data_loader):
        imgs=imgs.view(-1,config['Image']['image_size'])
        if use_gpu==True:
            imgs,labels=imgs.cuda(),labels.cuda()
        x_recon,mu,log_var=model(imgs)
        reconst_loss = F.binary_cross_entropy(x_recon, imgs, size_average=False)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss=reconst_loss+kl_div

        opt.zero_grad()
        loss.backward()
        opt.step()
        if (batch_id+1)%100==0:
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                   .format(epoch+1, max_epoch, batch_id+1, len(data_loader), reconst_loss.item(), kl_div.item()))
    if (epoch+1)%10==0:

        with torch.no_grad():
            # 保存采样图像，即潜在向量Z通过解码器生成的新图像
            z = torch.randn(config['Trainer']['batch_size'], config['Image']['z_dim']).cuda()
            out = model.decoder(z).view(-1, 1, 28, 28)
            sample_path=os.path.join(config['Solver']['output_dir'], 'sampled-epoch{}.png'.format(epoch+1))
            save_image(out, sample_path)
            print("{} has saved".format(sample_path))
            # 保存重构图像，即原图像通过解码器生成的图像
            out, _, _ = model(imgs)
            x_concat = torch.cat([imgs.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
            recon_path=os.path.join(config['Solver']['output_dir'], 'reconst-epoch{}.png'.format(epoch+1))
            save_image(x_concat, recon_path)
            print("{} has saved".format(recon_path))          
if __name__=='__main__':
    main()