import argparse


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser()

        # dataset related options 
        parser.add_argument('--dataset', type=str, default='kitti')
        parser.add_argument('--root', type=str, default='/hdd/local/sdb/umar/kitti_video/kitti_256')
        parser.add_argument('--frame_size', type=str, default='256 832')
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--shuffle', type=bool, default=False)
        parser.add_argument('--im_names_npy', type=str, default='kitti_im_names.npy')
        parser.add_argument('--read_frames_def', type=bool, help='True for kitti', default=True)
        parser.add_argument('--train', type=bool, default=True)

        # optimization related options 
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--beta2', type=float, default=0.999)

        # network related options 
        parser.add_argument('--disp_module', type=str, default='DispResNet')
        parser.add_argument('--pose_module', type=str, default='PoseResNet')
        parser.add_argument('--decoder_in_channels', type=int, default=2048)
        parser.add_argument('--disp_model_path', type=str, default=None)
        parser.add_argument('--pose_model_path', type=str, default=None)
        parser.add_argument('--learn_intrinsics', type=bool, default=False)

        # intermediate results realted options 
        parser.add_argument('--console_out', type=int, default=50)
        parser.add_argument('--save_disp', type=int, default=100)
        parser.add_argument('--log_tensorboard', type=bool, default=False)
        parser.add_argument('--int_results_dir', type=str, default='int_results/')
        parser.add_argument('--tboard_dir', type=str, default='tboard_dir/')
        parser.add_argument('--tboard_out', type=int, default=100)

        # saving the model 
        parser.add_argument('--save_model_dir', type=str, default='models/')
        parser.add_argument('--save_model_iter', type=int, default=5000)

        # training related options 
        parser.add_argument('--epochs', type=int, default=20)

        # gpus 
        parser.add_argument('--gpus', type=list, default=[3])

        # loss 
        parser.add_argument('--ssim_wt', type=float, default=0.85)
        parser.add_argument('--l1_wt', type=float, default=0.15)
        parser.add_argument('--smooth_wt', type=float, default=0.1)
        parser.add_argument('--geom_wt', type=float, default=0.5)
        parser.add_argument('--ssim_c1', type=float, default=0.01 ** 2)
        parser.add_argument('--ssim_c2', type=float, default=0.03 ** 2)

        self.opts = parser.parse_args()
        # changing the frame size 
        frame_size = self.opts.frame_size 
        frame_size = frame_size.split(' ')
        frame_size = [int(x) for x in frame_size]
        self.opts.frame_size = frame_size
        
        print(self.opts)

    def __call__(self):
        return self.opts



if __name__ == '__main__':
    Opts = Options()
    print(Opts.opts)

