import torch 
import torch.nn as nn 
import numpy as np 
import torchvision 
import importlib 
from test_options import Options 
import Datasets
import torch.utils.data as data 
import matplotlib.pyplot as plt 


class Test():
    def __init__(self, opts):
        self.opts = opts 
        self.model_path = opts.model_path 
        self.output_dir = opts.output_dir 
        self.gpu_id = opts.gpu_id 
        self.disp_module = opts.disp_module 
        self.dataset = opts.dataset 
        self.batch_size = opts.batch_size 
        self.train = opts.train 

        # The data loader 
        # getting the dataloader ready 
        if self.dataset == 'kitti':
            dataset = Datasets.KittiDataset(self.opts)
        elif self.dataset == 'nyu':
            dataset = Datasets.NYUDataset(self.opts)
        else:
            raise NameError('Dataset not found')
        self.DataLoader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, 
                                          num_workers=8)
        print('Data loader made')
        
        # loading the model 
        disp_module = importlib.import_module(self.disp_module)
        self.DispNet = disp_module.DispResNet()
        self.DispNet.load_state_dict(torch.load(self.model_path))
        if self.gpu_id is not None:
            self.device = torch.device('cuda:' + str(self.gpu_id[0]))
            self.DispNet = self.DispNet.to(self.device)
            if len(self.gpu_id) > 1:   
                self.DispNet = nn.DataParallel(self.DispNet, self.gpu_id)
        else:
            self.device = torch.device('cpu')
        self.DispNet.eval()
        print('Model Loaded')

        self.start_test()

    def start_test(self):
        print('Starting testing')
        with torch.no_grad():
            for i, batch_data in enumerate(self.DataLoader):
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                out_depth = 1.0 / self.DispNet(batch_data['curr_frame'])
                self.save_result(i, batch_data, out_depth)
                # for evaluation only 
                if i % 100 == 0:
                    print('Frames done {} out of {}'.format(self.batch_size * i,
                                                            self.batch_size * len(self.DataLoader)))
        print('finished')

    def save_result(self, i, batch_data, out_depth):
        b = batch_data['curr_frame'].size(0)
        for ii in range(b):
            curr_im_name = self.output_dir + ('%05d' % i) + '_' + str(ii) + '.png' 
            input_im = batch_data['curr_frame'][ii, :, :, :]
            depth = out_depth[ii, :, :, :] 
            depth_3 = self.gray2jet(depth)
            comb_im = torch.cat((input_im, depth_3), axis=-1)
            torchvision.utils.save_image(comb_im, curr_im_name)

    def gray2jet(self, dmap_0):
        cmap = plt.get_cmap('jet')
        dmap_0 = dmap_0[0, :, :].cpu().numpy()
        dmap_norm = (dmap_0 - dmap_0.min()) / (dmap_0.max() - dmap_0.min())
        dmap_col = cmap(dmap_norm)
        dmap_col = dmap_col[:, :, 0:3]
        dmap_col = np.transpose(dmap_col, (2, 0, 1))
        return torch.tensor(dmap_col).float().to(self.device)


if __name__ == '__main__':
    opts = Options().opts 
    Test(opts)


                

        
              
        
