import torch
import os
import torchvision.transforms as transforms 
import PIL.Image as Image 
import random
import matplotlib.pyplot as plt
import numpy as np
from options import Options


class KittiDataset():
    def __init__(self, opts):
        self.root = opts.root 
        self.frame_size = opts.frame_size  
        self.im_names_npy = opts.im_names_npy
        self.read_frames_def = opts.read_frames_def 
        self.train = opts.train 
        self.list_transforms = []
        if self.frame_size is not None:
            self.list_transforms.append(transforms.Resize(self.frame_size))
        self.list_transforms.append(transforms.ToTensor())
        self.list_transforms.append(transforms.Normalize([0.45, 0.45, 0.45],
                                                         [0.225, 0.225, 0.225]))
        self.transforms = transforms.Compose(self.list_transforms)
        # accessing sequences 
        self.sequences = []
        if self.train:
            seq_file = open(self.root + 'train.txt', 'r')
        else:
            seq_file = open(self.root + 'val.txt', 'r')
        for s in seq_file:
            seq_name = s.split('\n')[0]
            self.sequences += [seq_name]
        # acessing left videos
        self.curr_frames = []
        self.next_frames = []
        self.intrinsics = []
        self.inv_intrinsics = []

        if self.read_frames_def:
            self.read_frame_names_intrinsics()
    

    def read_frame_names_intrinsics(self):
        if not os.path.exists(self.im_names_npy):
            for s in self.sequences:
                frame_names = os.listdir(self.root + s + '/')
                if len(frame_names) < 3:
                    continue
                frame_names = [x for x in frame_names if x.endswith('.jpg') or x.endswith('.png')]
                frame_names.sort()
                sq_curr_frames = self.complete_name(frame_names[1:-1], s)
                sq_next_frames = self.complete_name(frame_names[2:], s)

                assert len(sq_curr_frames) == len(sq_next_frames)
                # getting the intrinsics 
                intr = np.genfromtxt(self.root + s + '/cam.txt').astype(np.float32).reshape(3, 3)
                inv_intr = np.linalg.inv(intr)
                intr_repeat = [intr] * len(sq_curr_frames)
                inv_intr_repeat = [inv_intr] * len(sq_curr_frames)

                self.curr_frames = self.curr_frames + sq_curr_frames
                self.next_frames = self.next_frames + sq_next_frames
                self.intrinsics = self.intrinsics + intr_repeat
                self.inv_intrinsics = self.inv_intrinsics + inv_intr_repeat
            self.data = {'curr_frames': self.curr_frames,
                    'next_frames': self.next_frames,
                    'intrinsics': self.intrinsics,
                    'inv_intrinsics': self.inv_intrinsics}
            np.save(self.im_names_npy, self.data) 
        else:
            self.data = np.load(self.im_names_npy, allow_pickle=True).item()
        assert len(self.data['curr_frames']) == \
                len(self.data['next_frames']) == \
                len(self.data['intrinsics']) == \
                len(self.data['inv_intrinsics']), print('Size mismatch')
    
    def complete_name(self, file_names, seq_name):
        return [self.root + seq_name + '/' + x for x in file_names]
        
    def display_sample(self):
        i = random.randint(0, len(self.data['curr_frames']))
        curr_frame = Image.open((self.data['curr_frames'])[i])
        next_frame = Image.open(self.data['next_frames'][i])
        intr = self.data['intrinsics'][i]
        inv_intr = self.data['inv_intrinsics'][i]

        print('Intrinsics {} and Inv intrinsics {}'.format(intr, inv_intr))
        
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(curr_frame)
        plt.subplot(2, 1, 2)
        plt.imshow(next_frame)
        plt.show()
    
    def __len__(self):
        return len(self.data['curr_frames'])

    def __getitem__(self, i):
        curr_frame = self.transforms(Image.open(self.data['curr_frames'][i]))
        next_frame = self.transforms(Image.open(self.data['next_frames'][i]))
        intr = torch.tensor(self.data['intrinsics'][i], requires_grad=False)
        inv_intr = torch.tensor(self.data['inv_intrinsics'][i], requires_grad=False)

        return {'curr_frame': curr_frame, 
                'next_frame': next_frame, 
                'intrinsics': intr,
                'intrinsics_inv': inv_intr}


class NYUDataset(KittiDataset):
    def __init__(self, opts):
        super().__init__(opts)
        self.sequences = []
        if self.train:
            seq_file = open(self.root + 'train.txt', 'r')
        else:
            seq_file = open(self.root + 'val.txt', 'r')

        for s in seq_file:
            seq_name = s.split('\n')[0]
            self.sequences += [seq_name]
        self.read_frame_names_intrinsics()

    def read_frame_names_intrinsics(self):
        if not os.path.exists(self.im_names_npy):
            for s in self.sequences:
                file_names = os.listdir(self.root + s + '/')
                if len(file_names) < 6:
                    continue
                cam_names = [x for x in file_names if x.endswith('cam.txt')]
                cam_names = sorted(cam_names)
                # reading the frame names 
                if len(cam_names) > 1:
                    frame_names = [x for  x in file_names if x.endswith('0.jpg') or x.endswith('0.png')]
                    frame_names_nxt = [x for x in file_names if x.endswith('1.jpg') or x.endswith('1.png')]
                else:
                    frame_names = [x for x in file_names if x.endswith('.jpg') or x.endswith('.png')] 
                    frame_names_nxt = frame_names.copy()
                frame_names = sorted(frame_names)
                frame_names_nxt = sorted(frame_names_nxt)
                sq_curr_frames = self.complete_name(frame_names, s)
                sq_next_frames = self.complete_name(frame_names_nxt, s)            

                # reading the intrinsics
                if len(cam_names) == 1:
                    intr = np.genfromtxt(self.root + s + '/cam.txt').astype(np.float32).reshape((3, 3))
                    inv_intr = np.linalg.inv(intr)
                    intr_repeat = [intr] * len(sq_curr_frames)
                    inv_intr_repeat = [inv_intr] * len(sq_curr_frames)
                else:
                    intr_repeat = []
                    inv_intr_repeat = []
                    for cam_name in cam_names:
                        full_cam_name = self.complete_name_cam(cam_name, s)
                        intr = np.genfromtxt(full_cam_name).astype(np.float32).reshape((3, 3)) 
                        inv_intr = np.linalg.inv(intr) 
                        intr_repeat += [intr] 
                        inv_intr_repeat += [inv_intr]     
                assert len(sq_curr_frames) == len(sq_next_frames) == \
                        len(intr_repeat) == len(inv_intr_repeat), print('Check folder: {}'.format(s))
                
                self.curr_frames = self.curr_frames + sq_curr_frames
                self.next_frames = self.next_frames + sq_next_frames
                self.intrinsics = self.intrinsics + intr_repeat
                self.inv_intrinsics = self.inv_intrinsics + inv_intr_repeat
            self.data = {'curr_frames': self.curr_frames,
                    'next_frames': self.next_frames,
                    'intrinsics': self.intrinsics,
                    'inv_intrinsics': self.inv_intrinsics}
            np.save(self.im_names_npy, self.data) 
        else:
            self.data = np.load(self.im_names_npy, allow_pickle=True).item()
        assert len(self.data['curr_frames']) == \
            len(self.data['next_frames']) == \
            len(self.data['intrinsics']) == \
            len(self.data['inv_intrinsics']), print('Size mismatch')
            
    def complete_name_cam(self, cam_name, seq_name):
        return self.root + seq_name + '/' + cam_name


if __name__ == '__main__':
    Opts = Options()
    dataset = NYUDataset(Opts.opts)
    dataset.display_sample()
    some_data = next(iter(dataset))
    print(some_data['curr_frame'].size())


