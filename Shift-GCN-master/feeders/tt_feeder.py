import numpy as np
import pickle
import torch
import pdb
from torch.utils.data import Dataset
import sys

sys.path.extend(['../'])
from feeders import tools


class Feeder(Dataset):
    def __init__(self, train_data_path, train_label_path, val_data_path, val_label_path, test_data_path, flag,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.flag = flag
        if self.flag == 'train':
            self.train_data_path = train_data_path
            self.train_label_path = train_label_path
            self.test_data_path = test_data_path
        elif self.flag == 'val':
            self.val_data_path = val_data_path
            self.val_label_path = val_label_path
        else:
            self.test_data_path = test_data_path
        
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
    
        self.use_mmap = use_mmap

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        if self.flag == 'train':
            try:
                with open(self.train_label_path) as f:
                    self.tr_sample_name, self.tr_label = pickle.load(f)
                with open(self.val_label_path) as f:
                    self.val_sample_name, self.val_label = pickle.load(f)
            except:
                # for pickle file from python2
                with open(self.train_label_path, 'rb') as f:
                    self.tr_sample_name, self.tr_label = pickle.load(f, encoding='latin1')
                with open(self.val_label_path, 'rb') as f:
                    self.val_sample_name, self.val_label = pickle.load(f, encoding='latin1')

            # self.sample_name = self.tr_sample_name + self.val_sample_name
            # self.label = self.tr_label + self.val_label
        # load data
        if self.use_mmap:
            if self.flag == 'train':
                self.tr_data = np.load(self.train_data_path, mmap_mode='r')
                self.val_data = np.load(self.val_data_path, mmap_mode='r')
                self.data = np.concatenate((self.tr_data, self.val_data), axis=0)
            else:
                self.data = np.load(self.test_data_path, mmap_mode='r')
        else:
            if self.flag == 'train':
                self.tr_data = np.load(self.train_data_path)
                self.val_data = np.load(self.val_data_path)
                self.data = np.concatenate((self.tr_data, self.val_data), axis=0)
            else:
                self.data = np.load(self.test_data_path, mmap_mode='r')
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]
        # self.data, self.label, self.aux_label = tools.sample_with_different_frequency(self.data, self.label)
        # pdb.set_trace()


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        if self.flag != 'test':
            label = self.label[index]
        data_numpy = np.array(data_numpy)
        data_numpy, aux_label = tools.dif_fre_transform(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        if self.flag != 'test':
            return data_numpy, label, aux_label, index
        else:
            return data_numpy, aux_label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path: 
    :param label_path: 
    :param vid: the id of sample
    :param graph: 
    :param is_3d: when vis NTU, set it True
    :return: 
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)


class TT_Feeder(Dataset):
    def __init__(self, train_data_path, train_label_path, val_data_path, val_label_path, test_data_path, flag,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.train_data_path = train_data_path
        self.train_label_path = train_label_path
        self.val_data_path = val_data_path
        self.val_label_path = val_label_path
        self.test_data_path = test_data_path
        
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.flag = flag
        self.use_mmap = use_mmap

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        if self.flag == 'train':
            try:
                with open(self.train_label_path) as f:
                    self.sample_name, self.label = pickle.load(f)
            except:
                with open(self.train_label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(f, encoding='latin1')
        elif self.flag == 'val':
            try:
                with open(self.val_label_path) as f:
                    self.sample_name, self.label = pickle.load(f)
            except:
                with open(self.val_label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(f, encoding='latin1')
            # self.sample_name = self.tr_sample_name + self.val_sample_name
            # self.label = self.tr_label + self.val_label
        # load data
        if self.use_mmap:
            if self.flag == 'train':
                self.train_data = np.load(self.train_data_path, mmap_mode='r')
                self.val_data = np.load(self.val_data_path, mmap_mode='r')
            elif self.flag == 'val':
                self.val_data = np.load(self.val_data_path, mmap_mode='r')
            else:
                self.test_data = np.load(self.test_data_path, mmap_mode='r')
        else:
            if self.flag == 'train':
                self.train_data = np.load(self.train_data_path)
                self.val_data = np.load(self.val_data_path)
            elif self.flag == 'val':
                self.val_data = np.load(self.val_data_path)
            else:
                self.test_data = np.load(self.test_data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]
        # self.data, self.label, self.aux_label = tools.sample_with_different_frequency(self.data, self.label)
        # pdb.set_trace()
        self.val_data_copy = self.val_data
       
        if self.flag == 'train':
            repeat_num = self.train_data.shape[0] // self.val_data.shape[0] + 1
            self.val_data = np.repeat(self.val_data, repeat_num, axis=0)
            self.val_data = self.val_data[:self.train_data.shape[0], :, :, :, :]
            self.train_data = self.train_data[:, :, :200, :, :]
            self.val_data = self.val_data[:, :, :200, :, :]
            # print(self.train_data.shape, self.test_data.shape)
        elif self.flag == 'val':
            self.val_data = self.val_data_copy[:, :, :200, :, :]
        else:
            self.test_data = self.test_data[:, :, :200, :, :]

        
        # pdb.set_trace()

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        if self.flag == 'train':
            return self.train_data.shape[0]
        elif self.flag == 'val':
            return self.val_data.shape[0]
        else:
            return self.test_data.shape[0]

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # data_numpy = None
        if self.flag == 'train':
            train_data_numpy = self.train_data[index]
            test_data_numpy = self.val_data[index]
            label = self.label[index]

            train_data_numpy = np.array(train_data_numpy)
            train_data_numpy, train_aux_label = tools.dif_fre_transform(train_data_numpy)

            test_data_numpy = np.array(test_data_numpy)
            test_data_numpy, test_aux_label = tools.dif_fre_transform(test_data_numpy)
        elif self.flag == 'val':
            val_data_numpy = self.val_data[index]
            val_data_numpy = np.array(val_data_numpy)
            val_data_numpy, val_aux_label = tools.dif_fre_transform(val_data_numpy)
            label = self.label[index]
        else:
            test_data_numpy = self.test_data[index]

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        if self.flag == 'train':
            return train_data_numpy, train_aux_label, label, test_data_numpy, test_aux_label, index
        elif self.flag == 'val':
            return label, val_data_numpy, val_aux_label, index
        else:
            return test_data_numpy, test_aux_label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

class TT_Feeder_v1(Dataset):
    def __init__(self, train_data_path, train_label_path, val_data_path, val_label_path, test_data_path, flag,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.train_data_path = train_data_path
        self.train_label_path = train_label_path
        self.val_data_path = val_data_path
        self.val_label_path = val_label_path
        self.test_data_path = test_data_path
        
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
    
        self.use_mmap = use_mmap

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        if self.flag != 'test':
            try:
                with open(self.train_label_path) as f:
                    self.tr_sample_name, self.tr_label = pickle.load(f)
                with open(self.val_label_path) as f:
                    self.val_sample_name, self.val_label = pickle.load(f)
            except:
                # for pickle file from python2
                with open(self.train_label_path, 'rb') as f:
                    self.tr_sample_name, self.tr_label = pickle.load(f, encoding='latin1')
                with open(self.val_label_path, 'rb') as f:
                    self.val_sample_name, self.val_label = pickle.load(f, encoding='latin1')

            # self.sample_name = self.tr_sample_name + self.val_sample_name
            # self.label = self.tr_label + self.val_label
        # load data
        if self.use_mmap:
            if self.flag != 'test':
                self.tr_data = np.load(self.train_data_path, mmap_mode='r')
                self.val_data = np.load(self.val_data_path, mmap_mode='r')
                # self.train_data = np.concatenate((self.tr_data, self.val_data), axis=0)
                # self.test_data = np.load(self.test_data_path, mmap_mode='r')
            else:
                self.data = np.load(self.test_data_path, mmap_mode='r')
        else:
            if self.flag != 'test':
                self.tr_data = np.load(self.train_data_path)
                self.val_data = np.load(self.val_data_path)
                # self.data = np.concatenate((self.tr_data, self.val_data), axis=0)
                # self.test_data = np.load(self.test_data_path)
            else:
                self.data = np.load(self.test_data_path, mmap_mode='r')
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]
        # self.data, self.label, self.aux_label = tools.sample_with_different_frequency(self.data, self.label)
        # pdb.set_trace()
        self.test_data = np.repeat(self.test_data, 4, axis=0)

        self.train_data = self.train_data[:, :, :200, :, :]
        self.test_data = self.test_data[:self.train_data.shape[0], :, :200, :, :]
        print(self.train_data.shape, self.test_data.shape)
        # pdb.set_trace()

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return self.data.shape[0]

    def __iter__(self):
        return self

    def __getitem__(self, index):
        train_data_numpy = self.train_data[index]
        test_data_numpy = self.test_data[index]
        if self.flag != 'test':
            label = self.label[index]
        train_data_numpy = np.array(data_numpy)
        train_data_numpy, train_aux_label = tools.dif_fre_transform(train_data_numpy)

        test_data_numpy = np.array(test_data_numpy)
        test_data_numpy, test_aux_label = tools.dif_fre_transform(test_data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        if self.flag != 'test':
            return train_data_numpy, train_aux_label, label, test_data_numpy, test_aux_label, index
        else:
            return test_data_numpy, test_aux_label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)