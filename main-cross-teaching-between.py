import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]

import time
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import transforms
import torch.distributed as dist
import math
import torchio
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from medpy.io import load,save
from tqdm import tqdm
from torchvision import utils
from hparam import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.nn.modules.loss import CrossEntropyLoss

source_train_dir = hp.source_train_dir
label_train_dir = hp.label_train_dir
unlabel_dir_train = hp.unlabel_dir



source_test_dir = hp.source_test_dir
label_test_dir = hp.label_test_dir

output_int_dir = hp.output_int_dir
output_float_dir = hp.output_float_dir


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def parse_training_args(parser):
    """	
    """

    parser.add_argument('-o', '--output_dir', type=str, default='logs', required=False, help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default='checkpoint_latest.pt', help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup') 

    training.add_argument('--epochs', type=int, default=20, help='Number of total epochs to run')   
    training.add_argument('--epochs-per-checkpoint', type=int, default=1, help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=1, help='batch-size')     #12
    training.add_argument('--sample', type=int, default=12, help='number of samples during training')    #12

    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )

    parser.add_argument("--init-lr", type=float, default=0.005, help="learning rate")   #0.001

    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    training.add_argument('--amp-run', action='store_true', help='Enable AMPa')
    training.add_argument('--cudnn-enabled', default=True, help='En	able cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true', help='disable uniform initialization of batchnorm layer weight')


    return parser



def train():

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark



    from data_function import MedData_train,Med_unlabel_train,My_unlabel
    os.makedirs(args.output_dir, exist_ok=True)

    if hp.mode == '2d':
        #from models.two_d.unet import Unet
        #model = Unet(in_channels=hp.in_class, classes=hp.out_class)

        from models.two_d.miniseg import MiniSeg
        model = MiniSeg(in_input=hp.in_class, classes=hp.out_class)

        # from models.two_d.fcn import FCN32s as fcn
        # model = fcn(in_class =hp.in_class,n_class=hp.out_class)

        # from models.two_d.segnet import SegNet
        # model = SegNet(input_nbr=hp.in_class,label_nbr=hp.out_class)

        #from models.two_d.deeplab import DeepLabV3
        #model = DeepLabV3(in_class=hp.in_class,class_num=hp.out_class)

        # from models.two_d.unetpp import ResNet34UnetPlus
        # model = ResNet34UnetPlus(num_channels=hp.in_class,num_class=hp.out_class)

        # from models.two_d.pspnet import PSPNet
        # model = PSPNet(in_class=hp.in_class,n_classes=hp.out_class)


    elif hp.mode == '3d':
        # from models.three_d.unet3d import UNet3D
        # model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=8)#, base_n_filter=2)  #2
        
        # from models.three_d.Residual_unet3d import UNet
        # model = UNet(in_channels=hp.in_class, n_classes=hp.out_class, base_n_filter=4) #1
        
        # from models.three_d.gaborunet import UNet3D   #8->11   4->
        # model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=16)#, base_n_filter=2)

        #from models.three_d.fcn3d import FCN_Net
        #model = FCN_Net(in_channels =hp.in_class,n_class =hp.out_class)

        # from models.three_d.highresnet import HighRes3DNet
        # model = HighRes3DNet(in_channels=hp.in_class,out_channels=hp.out_class)

        # from models.three_d.densenet3d import SkipDenseNet3D
        # model = SkipDenseNet3D(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.densevoxelnet3d import DenseVoxelNet  #1
        # model = DenseVoxelNet(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.vnet3d import VNet   #2
        # model = VNet(in_channels=hp.in_class, classes=hp.out_class)
        
        # from models.three_d.unetr import UNETR
        # model = UNETR(img_shape=(hp.crop_or_pad_size,hp.crop_or_pad_size,hp.crop_or_pad_size), input_dim=hp.in_class, output_dim=hp.out_class)

        from models.three_d.chen_trans import UNETR
        model = UNETR(img_shape=(hp.crop_or_pad_size,hp.crop_or_pad_size,hp.crop_or_pad_size), input_dim=hp.in_class, output_dim=hp.out_class)

        model_trans = UNETR(img_shape=(hp.crop_or_pad_size,hp.crop_or_pad_size,hp.crop_or_pad_size), input_dim=hp.in_class, output_dim=hp.out_class)





    model = torch.nn.DataParallel(model)
    model_trans = torch.nn.DataParallel(model_trans)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    optimizer_trans = torch.optim.Adam(model_trans.parameters(), lr=args.init_lr)

    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.8)
    scheduler_trans = StepLR(optimizer_trans, step_size=30, gamma=0.8)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        # scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    model.cuda()

    from loss_function import Binary_Loss,MSE_Loss,weightbceloss1,BinaryDiceLoss
    criterion = Binary_Loss().cuda()
    criterion_1 = weightbceloss1().cuda()

    # ce_loss = CrossEntropyLoss()
    ce_loss = criterion
    # dice_loss = DiceLoss(1)
    dice_loss = BinaryDiceLoss()


    writer = SummaryWriter(args.output_dir)
    
    train_unlabel_dataset = Med_unlabel_train(unlabel_dir_train)
    # train_unlabel_dataset = Med_unlabel_train(source_train_dir)



    my_unlabel = My_unlabel(train_unlabel_dataset.queue_dataset,train_unlabel_dataset.queue_dataset_2)
    # train_unlabel_loader = DataLoader(train_unlabel_dataset.queue_dataset, 
    #                         batch_size=args.batch, 
    #                         shuffle=False,
    #                         pin_memory=True,
    #                         drop_last=True)

    # train_unlabel_loader_unaug = DataLoader(train_unlabel_dataset.queue_dataset_2, 
    #                         batch_size=args.batch, 
    #                         shuffle=False,
    #                         pin_memory=True,
    #                         drop_last=True)


    train_unlabel_loader = DataLoader(my_unlabel, 
                            batch_size=args.batch, 
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

    train_dataset = MedData_train(source_train_dir,label_train_dir)
    train_loader = DataLoader(train_dataset.queue_dataset, 
                            batch_size=args.batch, 
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

    model.train()

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)



    for epoch in range(1, epochs + 1):
        print("epoch:"+str(epoch))
        epoch += elapsed_epochs

        train_epoch_avg_loss = 0.0
        num_iters = 0


        for i, batch in enumerate(train_loader):
            
            
            # random_value = np.random.randint(0,len(train_unlabel_loader))
            # for jj,(aug,un_aug) in enumerate(zip(train_unlabel_loader,train_unlabel_loader_unaug)):
            #     unlabel_img = aug['source']['data']
            #     unlabel_img_unaug = un_aug['source']['data']

                
            #     if jj == random_value:
            #         break

            for data_unlabel,data_unlabel_2 in train_unlabel_loader:

                unlabel_img = data_unlabel['source']['data']
                unlabel_img_unaug = data_unlabel_2['source']['data']   
                break   #

            if hp.debug:
                if i >=1:
                    break

            print(f'Batch: {i}/{len(train_loader)} epoch {epoch}')

            optimizer.zero_grad()


            if (hp.in_class == 1) and (hp.out_class == 1) :
                x = batch['source']['data']
                y = batch['label']['data']

                x = x.type(torch.FloatTensor).cuda()
                y = y.type(torch.FloatTensor).cuda()


            else:
                x = batch['source']['data']
                y_atery = batch['atery']['data']
                y_lung = batch['lung']['data']
                y_trachea = batch['trachea']['data']
                y_vein = batch['atery']['data']

                x = x.type(torch.FloatTensor).cuda()

                y = torch.cat((y_atery,y_lung,y_trachea,y_vein),1) 
                y = y.type(torch.FloatTensor).cuda()


            if hp.mode == '2d':
                x = x.squeeze(4)
                y = y.squeeze(4)

                y = y/255.

            # print(y.max())



            #################
            outputs = model(x)
            outputs_soft = torch.sigmoid(outputs)
            output_trans = model_trans(x)
            output_soft_trans = torch.sigmoid(output_trans)

            outputs_un = model(unlabel_img)
            outputs_soft_un = torch.sigmoid(outputs_un)
            output_trans_un = model_trans(unlabel_img)
            output_soft_trans_un = torch.sigmoid(output_trans_un)


            consistency_weight = math.exp(-0.1*(20-epoch)**1.5)


            loss1 = 0.5 * (ce_loss(outputs, y) + dice_loss(
                outputs_soft, y))
            loss2 = 0.5 * (ce_loss(output_trans, y) + dice_loss(
                output_soft_trans, y))

            pseudo_outputs1 = (outputs_soft_un.detach()>0.5).float()
            pseudo_outputs2 = (output_soft_trans_un.detach()>0.5).float()

            pseudo_supervision1 = dice_loss(
                outputs_soft_un, pseudo_outputs2)
            pseudo_supervision2 = dice_loss(
                output_soft_trans_un, pseudo_outputs1)

            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2

            loss = model1_loss + model2_loss


            
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #################

            # for metrics
            logits = torch.sigmoid(outputs)
            labels = logits.clone()
            labels[labels>0.5] = 1
            labels[labels<=0.5] = 0

            num_iters += 1
            iteration += 1


            false_positive_rate,false_negtive_rate,dice = metric(y.cpu(),labels.cpu())
            ## log
            writer.add_scalar('Training/Loss', loss.item(),iteration)
            writer.add_scalar('Training/false_positive_rate', false_positive_rate,iteration)
            writer.add_scalar('Training/false_negtive_rate', false_negtive_rate,iteration)
            writer.add_scalar('Training/dice', dice,iteration)
            


            print("loss:"+str(loss.item()))
            print('lr:'+str(scheduler._last_lr[0]))

            

        scheduler.step()


        # Store latest checkpoint in each epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "epoch": epoch,

            },
            os.path.join(args.output_dir, args.latest_checkpoint_file),
        )




        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0:

            torch.save(
                {
                    
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt"),
            )
        


            
            with torch.no_grad():
                if hp.mode == '2d':
                    x = x.unsqueeze(4)
                    y = y.unsqueeze(4)
                    outputs = outputs.unsqueeze(4)


                x = x[0].cpu().detach().numpy()
                y = y[0].cpu().detach().numpy()
                outputs = outputs[0].cpu().detach().numpy()
                affine = batch['source']['affine'][0].numpy()

                if (hp.in_class == 1) and (hp.out_class == 1) :
                    source_image = torchio.ScalarImage(tensor=x, affine=affine)
                    source_image.save(os.path.join(args.output_dir,("step-{}-source.mhd").format(epoch)))

                    label_image = torchio.ScalarImage(tensor=y, affine=affine)
                    label_image.save(os.path.join(args.output_dir,("step-{}-gt.mhd").format(epoch)))

                    output_image = torchio.ScalarImage(tensor=outputs, affine=affine)
                    output_image.save(os.path.join(args.output_dir,("step-{}-predict.mhd").format(epoch)))
                else:
                    y = np.expand_dims(y, axis=1)
                    outputs = np.expand_dims(outputs, axis=1)

                    source_image = torchio.ScalarImage(tensor=x, affine=affine)
                    source_image.save(os.path.join(args.output_dir,("step-{}-source.mhd").format(epoch)))

                    label_image_artery = torchio.ScalarImage(tensor=y[0], affine=affine)
                    label_image_artery.save(os.path.join(args.output_dir,("step-{}-gt_artery.mhd").format(epoch)))

                    output_image_artery = torchio.ScalarImage(tensor=outputs[0], affine=affine)
                    output_image_artery.save(os.path.join(args.output_dir,("step-{}-predict_artery.mhd").format(epoch)))

                    label_image_lung = torchio.ScalarImage(tensor=y[1], affine=affine)
                    label_image_lung.save(os.path.join(args.output_dir,("step-{}-gt_lung.mhd").format(epoch)))

                    output_image_lung = torchio.ScalarImage(tensor=outputs[1], affine=affine)
                    output_image_lung.save(os.path.join(args.output_dir,("step-{}-predict_lung.mhd").format(epoch)))

                    label_image_trachea = torchio.ScalarImage(tensor=y[2], affine=affine)
                    label_image_trachea.save(os.path.join(args.output_dir,("step-{}-gt_trachea.mhd").format(epoch)))

                    output_image_trachea = torchio.ScalarImage(tensor=outputs[2], affine=affine)
                    output_image_trachea.save(os.path.join(args.output_dir,("step-{}-predict_trachea.mhd").format(epoch)))

                    label_image_vein = torchio.ScalarImage(tensor=y[3], affine=affine)
                    label_image_vein.save(os.path.join(args.output_dir,("step-{}-gt_vein.mhd").format(epoch)))

                    output_image_vein = torchio.ScalarImage(tensor=outputs[3], affine=affine)
                    output_image_vein.save(os.path.join(args.output_dir,("step-{}-predict_vein.mhd").format(epoch)))           


    writer.close()


def test():

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Testing')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from data_function import MedData_test

    os.makedirs(output_float_dir, exist_ok=True)
    os.makedirs(output_int_dir, exist_ok=True)

    if hp.mode == '2d':
        #from models.two_d.unet import Unet
        #model = Unet(in_channels=hp.in_class, classes=hp.out_class)

        from models.two_d.miniseg import MiniSeg
        model = MiniSeg(in_input=hp.in_class, classes=hp.out_class)

        # from models.two_d.fcn import FCN32s as fcn
        # model = fcn(in_class =hp.in_class,n_class=hp.out_class)

        # from models.two_d.segnet import SegNet
        # model = SegNet(input_nbr=hp.in_class,label_nbr=hp.out_class)

        #from models.two_d.deeplab import DeepLabV3
        #model = DeepLabV3(in_class=hp.in_class,class_num=hp.out_class)

        # from models.two_d.unetpp import ResNet34UnetPlus
        # model = ResNet34UnetPlus(num_channels=hp.in_class,num_class=hp.out_class)

        # from models.two_d.pspnet import PSPNet
        # model = PSPNet(in_class=hp.in_class,n_classes=hp.out_class)

    elif hp.mode == '3d':
        # from models.three_d.unet3d import UNet3D
        # model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=8)#, base_n_filter=2)
        
        # from models.three_d.Residual_unet3d import UNet
        # model = UNet(in_channels=hp.in_class, n_classes=hp.out_class, base_n_filter=4) 
        
        # from models.three_d.gaborunet import UNet3D
        # model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class, init_features=16)#, base_n_filter=2)

        # from models.three_d.fcn3d import FCN_Net
        # model = FCN_Net(in_channels =hp.in_class,n_class =hp.out_class)

        # from models.three_d.highresnet import HighRes3DNet
        # model = HighRes3DNet(in_channels=hp.in_class,out_channels=hp.out_class)

        # from models.three_d.densenet3d import SkipDenseNet3D
        # model = SkipDenseNet3D(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.densevoxelnet3d import DenseVoxelNet
        # model = DenseVoxelNet(in_channels=hp.in_class, classes=hp.out_clas

        # from models.three_d.vnet3d import VNet
        # model = VNet(in_channels=hp.in_class, classes=hp.out_class)

        # from models.three_d.unetr import UNETR
        # model = UNETR(img_shape=(hp.crop_or_pad_size,hp.crop_or_pad_size,hp.crop_or_pad_size), input_dim=hp.in_class, output_dim=hp.out_class)

        from models.three_d.chen_trans import UNETR
        model = UNETR(img_shape=(hp.crop_or_pad_size,hp.crop_or_pad_size,hp.crop_or_pad_size), input_dim=hp.in_class, output_dim=hp.out_class)




    model = torch.nn.DataParallel(model)


    print("load model:", args.ckpt)
    print(os.path.join(args.output_dir, args.latest_checkpoint_file))
    ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"])


    model.cuda()
    model.eval()


    test_dataset = MedData_test(source_test_dir,label_test_dir)
    znorm = ZNormalization()

    if hp.mode == '3d':
        patch_overlap = 4,4,4
        patch_size = hp.patch_size,hp.patch_size,hp.patch_size
        # patch_size = 128, 128, 64
    elif hp.mode == '2d':
        patch_overlap = 4,4,0
        patch_size = hp.patch_size,hp.patch_size,1


    for i,subj in enumerate(test_dataset.subjects):
        subj = znorm(subj)
        grid_sampler = torchio.inference.GridSampler(
                subj,
                patch_size,
                patch_overlap,
            )

        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
        aggregator = torchio.inference.GridAggregator(grid_sampler)
        aggregator_1 = torchio.inference.GridAggregator(grid_sampler)
        
        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):


                input_tensor = patches_batch['source'][torchio.DATA].to(device)
                locations = patches_batch[torchio.LOCATION]

                if hp.mode == '2d':
                    input_tensor = input_tensor.squeeze(4)
                outputs = model(input_tensor)

                if hp.mode == '2d':
                    outputs = outputs.unsqueeze(4)
                logits = torch.sigmoid(outputs)

                labels = logits.clone()
                labels[labels>0.5] = 1
                labels[labels<=0.5] = 0

                aggregator.add_batch(logits, locations)
                aggregator_1.add_batch(labels, locations)
        output_tensor = aggregator.get_output_tensor()
        output_tensor_1 = aggregator_1.get_output_tensor()




        affine = subj['source']['affine']
        if (hp.in_class == 1) and (hp.out_class == 1) :
            if i<9:
                label_image = torchio.ScalarImage(tensor=output_tensor.numpy(), affine=affine)
                label_image.save(os.path.join(output_float_dir,"0"+str(i+1)+".mhd"))
                output_image = torchio.ScalarImage(tensor=output_tensor_1.numpy(), affine=affine)
                output_image.save(os.path.join(output_int_dir,"0"+str(i+1)+".mhd"))
            else:
                label_image = torchio.ScalarImage(tensor=output_tensor.numpy(), affine=affine)
                label_image.save(os.path.join(output_float_dir,str(i+1)+".mhd"))
                output_image = torchio.ScalarImage(tensor=output_tensor_1.numpy(), affine=affine)
                output_image.save(os.path.join(output_int_dir,str(i+1)+".mhd"))
        else:
            output_tensor = output_tensor.unsqueeze(1)
            output_tensor_1= output_tensor_1.unsqueeze(1)

            output_image_artery_float = torchio.ScalarImage(tensor=output_tensor[0].numpy(), affine=affine)
            output_image_artery_float.save(os.path.join(output_int_dir,str(i)+"_result_float_artery.mhd"))

            output_image_artery_int = torchio.ScalarImage(tensor=output_tensor_1[0].numpy(), affine=affine)
            output_image_artery_int.save(os.path.join(output_int_dir,str(i)+"_result_int_artery.mhd"))

            output_image_lung_float = torchio.ScalarImage(tensor=output_tensor[1].numpy(), affine=affine)
            output_image_lung_float.save(os.path.join(output_int_dir,str(i)+"_result_float_lung.mhd"))

            output_image_lung_int = torchio.ScalarImage(tensor=output_tensor_1[1].numpy(), affine=affine)
            output_image_lung_int.save(os.path.join(output_int_dir,str(i)+"_result_int_lung.mhd"))

            output_image_trachea_float = torchio.ScalarImage(tensor=output_tensor[2].numpy(), affine=affine)
            output_image_trachea_float.save(os.path.join(output_int_dir,str(i)+"_result_float_trachea.mhd"))

            output_image_trachea_int = torchio.ScalarImage(tensor=output_tensor_1[2].numpy(), affine=affine)
            output_image_trachea_int.save(os.path.join(output_int_dir,str(i)+"_result_int_trachea.mhd"))

            output_image_vein_float = torchio.ScalarImage(tensor=output_tensor[3].numpy(), affine=affine)
            output_image_vein_float.save(os.path.join(output_int_dir,str(i)+"_result_float_veiny.mhd"))

            output_image_vein_int = torchio.ScalarImage(tensor=output_tensor_1[3].numpy(), affine=affine)
            output_image_vein_int.save(os.path.join(output_int_dir,str(i)+"_result_int_vein.mhd"))           


   

if __name__ == '__main__':
    if hp.train_or_test == 'train':
        train()
    elif hp.train_or_test == 'test':
        test()
