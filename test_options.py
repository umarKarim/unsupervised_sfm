import argparse 

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--root', type=str, default='/hdd/local/sdb/umar/kitti_video/kitti_256/')
        parser.add_argument('--dataset', type=str, default='kitti')
        parser.add_argument('--frame_size', type=str, default="256 832")
        parser.add_argument('--shuffle', type=bool, default=False)
        parser.add_argument('--im_names_npy', type=str, default='kitti_test_im_names.npy')
        parser.add_argument('--read_frames_def', type=bool, help='True for kitti', default=True)
        parser.add_argument('--batch_size', type=int, default=1)

        parser.add_argument('--output_dir', type=str, default='./output_results/')
        parser.add_argument('--disp_module', type=str, default='DispResNet')
        parser.add_argument('--model_path', type=str, default='nyu_models/Disp_018_03788.pth')
        parser.add_argument('--gpu_id', type=int, default=[2])
        parser.add_argument('--train', type=bool, default=False)

        self.opts = parser.parse_args() 
        frame_size = self.opts.frame_size 
        self.opts.frame_size = [int(x) for x in frame_size.split(' ')]

        print(self.opts)
        
    def __call__(self):
        return self.opts 


