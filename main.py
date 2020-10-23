import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
import torchvision.utils as vutils
import torch.utils.tensorboard as tb 
import torch.utils.data as data 
import Datasets
import importlib
from Loss import Loss 
from options import Options
import time 
import matplotlib.pyplot as plt
import numpy as np 



class MainTrain():
    def __init__(self, opts):
        self.opts = opts 
        self.epochs = opts.epochs 
        self.batch_size = opts.batch_size
        self.shuffle = opts.shuffle 
        self.lr = opts.lr
        self.beta1 = opts.beta1 
        self.beta2 = opts.beta2
        self.console_out = opts.console_out 
        self.save_disp = opts.save_disp 
        self.tboard_out = opts.tboard_out 
        self.log_tb = opts.log_tensorboard 
        self.disp_module = opts.disp_module 
        self.pose_module = opts.pose_module 
        self.dataset = opts.dataset
        self.gpus = opts.gpus  
        self.tboard_dir = opts.tboard_dir 
        self.int_result_dir = opts.int_results_dir 
        self.save_model_dir = opts.save_model_dir 
        self.save_model_iter = opts.save_model_iter 
        self.frame_size = opts.frame_size
        self.disp_model_path = opts.disp_model_path 
        self.pose_model_path = opts.pose_model_path
        self.start_time = time.time()

        # getting the dataloader ready 
        if self.dataset == 'kitti':
            dataset = Datasets.KittiDataset(self.opts)
        elif self.dataset == 'nyu':
            dataset = Datasets.NYUDataset(self.opts)
        else:
            raise NameError('Dataset not found')
        self.DataLoader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, 
                                          num_workers=8)
        
        # loading the modules 
        if len(self.gpus) == 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:' + str(self.gpus[0]))
        disp_module = importlib.import_module(self.disp_module)
        pose_module = importlib.import_module(self.pose_module)
        self.DispModel = disp_module.DispResNet().to(self.device)
        self.PoseModel = pose_module.PoseResNet().to(self.device)
        if self.disp_model_path is not None:
            self.DispModel.load_state_dict(torch.load(self.disp_model_path))
        if self.pose_model_path is not None:
            self.PoseModel.load_state_dict(torch.load(self.pose_model_path))
        if len(self.gpus) != 0:
            self.DispModel = nn.DataParallel(self.DispModel, self.gpus)
            self.PoseModel = nn.DataParallel(self.PoseModel, self.gpus)
        self.DispModel.to(self.device)
        self.PoseModel.to(self.device)
        self.Loss = Loss(opts)

        # the optimizer 
        params_dict = [{'params': self.DispModel.parameters()},
                       {'params': self.PoseModel.parameters()}]
        self.optim = torch.optim.Adam(params_dict, lr=self.lr, betas=[self.beta1, self.beta2])

        # data logger and output
        self.fix_batch = next(iter(self.DataLoader))
        self.writer = tb.SummaryWriter(self.tboard_dir)

        self.start()

    def start(self):
        for epoch in range(self.epochs):
            for i, in_data in enumerate(self.DataLoader):
                for key in in_data.keys():
                    in_data[key] = in_data[key].to(self.device)
                out_depth = self.get_depth(in_data) 
                out_pose = self.get_poses(in_data)
                
                losses, net_loss = self.Loss(in_data, out_depth, out_pose)
                self.optim.zero_grad()
                net_loss.backward()
                self.optim.step() 

                self.console_display(epoch, i, losses, out_depth, out_pose)
                int_disp = self.save_int_result(epoch, i, out_depth, in_data)
                self.save_model(epoch, i) 
                self.save_tboard(epoch, i, losses, int_disp)

    def get_depth(self, in_data):
        out_depth = {}
        out_depth['curr_depth'] = 1.0 / self.DispModel(in_data['curr_frame'])
        out_depth['next_depth'] = 1.0 / self.DispModel(in_data['next_frame'])
        return out_depth

    def get_poses(self, in_data):
        out_pose = {}
        pose_net_out_for, foci, offsets = self.PoseModel(in_data['curr_frame'], in_data['next_frame'])
        out_pose['curr2nxt'] = pose_net_out_for[:, :6]
        intr_for = self.correct_intrinsics(foci, offsets)
        # intr_for = pose_net_out_for[:, 6:]
        pose_net_out_rev, foci, offsets = self.PoseModel(in_data['next_frame'], in_data['curr_frame'])
        out_pose['curr2nxt_inv'] = pose_net_out_rev[:, :6]
        intr_rev = self.correct_intrinsics(foci, offsets)
        # intr_rev = pose_net_out_rev[:, 6:]
        intr = (intr_for + intr_rev) / 2.0
        out_pose['intrinsics'] = intr.view(-1, 3, 3)
        # out_pose['curr2nxt'] = self.PoseModel(in_data['curr_frame'], in_data['next_frame'])
        # out_pose['curr2nxt_inv'] = self.PoseModel(in_data['next_frame'], in_data['curr_frame'])
        return out_pose

    def correct_intrinsics(self, foci, offsets):
        h, w = self.frame_size[0], self.frame_size[1]
        b = foci.size(0)
        # applying corrections based on frame size (taken from depth from videos from wild)
        foci_out = foci.clone() 
        foci_out[:, 0] = foci[:, 0] * w 
        foci_out[:, 1] = foci[:, 1] * h 
        offsets_out = offsets.clone()  
        offsets_out[:, 0] = (offsets_out[:, 0] + 0.5) * w 
        offsets_out[:, 1] = (offsets_out[:, 1] + 0.5) * h 
        ones = torch.ones(b).to(self.device)
        zeros = torch.zeros(b).to(self.device)
        intr = torch.empty((b, 3, 3), requires_grad=False).to(self.device)
        intr[: ,0, 0] = foci_out[:, 0] 
        intr[:, 0, 1] = zeros.clone() 
        intr[:, 0, 2] = offsets_out[:, 0] 
        intr[:, 1, 0] = zeros.clone() 
        intr[:, 1, 1] = foci_out[:, 1]
        intr[:, 1, 2] = offsets_out[:, 1]
        intr[:, 2, 0] = zeros.clone() 
        intr[:, 2, 1] = zeros.clone() 
        intr[:, 2, 2] = ones.clone()
        return intr

    def console_display(self, epoch, i, losses, out_depth, out_pose):
        if i % self.console_out == 0:
            loss_list = ''
            for key, loss in losses.items():
                loss_list = loss_list + key + ':' + str(loss.item()) + ', ' 
            tt = time.time() - self.start_time
            tot_b = len(self.DataLoader)
            print('Epoch: {}, batch: {} out of {}, time: {}'.format(epoch, i, tot_b, tt))
            print(loss_list)

            depth_mean = torch.mean(out_depth['curr_depth'])
            depth_min = torch.min(out_depth['curr_depth'])
            depth_max = torch.max(out_depth['curr_depth'])
            pose_mean = torch.mean(out_pose['curr2nxt'])
            pose_min = torch.min(out_pose['curr2nxt'])
            pose_max = torch.max(out_pose['curr2nxt'])
            print('Mean depth: {}, max depth: {}, min depth: {}'.format(depth_mean.item(),
                                                                        depth_max.item(), 
                                                                        depth_min.item()))
            print('Mean pose: {}, max pose: {}, min pose: {}'.format(pose_mean.item(),
                                                                     pose_max.item(), 
                                                                     pose_min.item()))
            
    def save_int_result(self, epoch, i, out_depth, in_data):
        if i % self.save_disp == 0:
            with torch.no_grad():
                fix_disp = out_depth['curr_depth']
                auto_mask = self.Loss.auto_mask 
                depth_mask = self.Loss.depth_mask 
                valid_mask = self.Loss.valid_mask.float() 
                recon_fr = self.Loss.recon_tar_fr 
                fix_disp = self.gray2jet(fix_disp)
                auto_mask = self.from1ch23ch(auto_mask)
                depth_mask = self.from1ch23ch(depth_mask)
                valid_mask = self.from1ch23ch(valid_mask) 
                recon_fr = self.from1ch23ch(recon_fr)
                fix_im = (in_data['curr_frame'])[0]
                
                hor_im1 = torch.cat((fix_im, recon_fr), dim=-1)
                hor_im2 = torch.cat((valid_mask, auto_mask), dim=-1)
                hor_im3 = torch.cat((depth_mask, fix_disp), dim=-1)
                comb_im = torch.cat((hor_im1, hor_im2, hor_im3), dim=1)
                # comb_im = torch.cat((fix_im, fix_disp), dim=-1)
                ep_str = ('%03d_' % epoch)
                iter_str = ('%05d' % i)
                im_name = self.int_result_dir + ep_str + iter_str + '.png'
                vutils.save_image(comb_im, im_name)
            print('Intermediate result saved')
            return fix_disp
    
    def from1ch23ch(self, im):
        assert len(im.size()) == 4
        new_im = im
        if im.size(1) == 1:
            new_im = im / im.max() 
            new_im = torch.cat((new_im, new_im, new_im), dim=1)
        new_im = new_im[0, :, :, :]
        return new_im

    def gray2jet(self, dmap):
        cmap = plt.get_cmap('jet')
        dmap_0 = dmap[0, 0, :, :].cpu().numpy()
        dmap_norm = (dmap_0 - dmap_0.min()) / (dmap_0.max() - dmap_0.min())
        dmap_col = cmap(dmap_norm)
        dmap_col = dmap_col[:, :, 0:3]
        dmap_col = np.transpose(dmap_col, (2, 0, 1))
        return torch.tensor(dmap_col).float().to(self.device)

    def save_tboard(self, epoch, i, losses, int_disp):
        global_step = epoch * len(self.DataLoader) + i
        if i % self.tboard_out and self.log_tb:
            for key, loss in losses.items():
                self.writer.add_scalar('Loss' + key, loss, global_step)
            self.writer.add_image('Disparity', int_disp, global_step)
    
    def save_model(self, epoch, i):
        global_step = epoch * len(self.DataLoader) + i 
        if global_step % self.save_model_iter == 0:
            ep_str = ('%03d_' % epoch)
            iter_str = ('%05d' % i)
            disp_model_name = self.save_model_dir + 'Disp_' + ep_str + iter_str + '.pth'
            pose_model_name = self.save_model_dir + 'Pose_' + ep_str + iter_str + '.pth'
            torch.save(self.DispModel.module.state_dict(), disp_model_name)
            torch.save(self.PoseModel.module.state_dict(), pose_model_name)
            print('Model saved')



if __name__ == '__main__':
    Opts = Options()
    OnlineDepth = MainTrain(Opts.opts)
