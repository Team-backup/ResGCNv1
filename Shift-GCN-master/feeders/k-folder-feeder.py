class UAV_Feeder1(Dataset):
    def __init__(self, phase, path, data_shape, connect_joint, debug, **kwargs):
        _, _, self.T, self.V, self.M = data_shape

        self.conn = connect_joint
        data_path = '{}/train_data.npy'.format(path)
        label_path = '{}/train_label.pkl'.format(path)

        if os.path.exists(data_path) and os.path.exists(label_path):
            self.data = np.load(data_path, mmap_mode='r')
            with open(label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        else:
            logging.info('')
            logging.error('Error: Do NOT exist data files: {} or {}!'.format(data_path, label_path))
            logging.info('Please generate data first!')
            raise ValueError()

        self.data = self.data[:,:,:200,:,:]

        index1 = np.load('/home/data1_4t/dpx/ResGCNv3/L1.npy')
        index2 = np.load('/home/data1_4t/dpx/ResGCNv3/L2.npy')
        index3 = np.load('/home/data1_4t/dpx/ResGCNv3/L3.npy')

        if phase=='train':

            train_index = np.concatenate((index2,index3),axis=0).tolist()
            self.data = self.data[train_index,:,:,:,:]
            self.label = [self.label[i] for i in train_index]
            self.sample_name = [self.sample_name[i] for i in train_index]

        else:

            eval_index = index1.tolist()
            self.data = self.data[eval_index,:,:,:,:]
            self.label = [self.label[i] for i in eval_index]
            self.sample_name = [self.sample_name[i] for i in eval_index]



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        label = self.label[idx]
        name = self.sample_name[idx]
        # subjs = self.subjs[idx]
        # print(label)

        # (C, T, V, M) -> (I, C*2, T, V, M)
        # T 帧数， V 关节点个数，M人的个数，C ：xyz坐标，这个函数目的输入6个输入
        # 绝对位置，相对位置，一阶速度，二阶速度，关节长度，关节角度
        # 所以这里的 C*2 = 6， I=3，等价于之前的C，表示关节点坐标
        data = multi_input(data, self.conn)

        return data, label, name

class UAV_Feeder2(Dataset):
    def __init__(self, phase, path, data_shape, connect_joint, debug, **kwargs):
        _, _, self.T, self.V, self.M = data_shape

        self.conn = connect_joint
        data_path = '{}/train_data.npy'.format(path)
        label_path = '{}/train_label.pkl'.format(path)

        if os.path.exists(data_path) and os.path.exists(label_path):
            self.data = np.load(data_path, mmap_mode='r')
            with open(label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        else:
            logging.info('')
            logging.error('Error: Do NOT exist data files: {} or {}!'.format(data_path, label_path))
            logging.info('Please generate data first!')
            raise ValueError()

        self.data = self.data[:,:,:200,:,:]

        index1 = np.load('/home/data1_4t/dpx/ResGCNv3/L1.npy')
        index2 = np.load('/home/data1_4t/dpx/ResGCNv3/L1.npy')
        index3 = np.load('/home/data1_4t/dpx/ResGCNv3/L1.npy')

        if phase=='train':

            train_index = np.concatenate((index1,index3),axis=0).tolist()
            self.data = self.data[train_index,:,:,:,:]
            self.label = [self.label[i] for i in train_index]
            self.sample_name = [self.sample_name[i] for i in train_index]

        else:

            eval_index = index2.tolist()
            self.data = self.data[eval_index,:,:,:,:]
            self.label = [self.label[i] for i in eval_index]
            self.sample_name = [self.sample_name[i] for i in eval_index]



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        label = self.label[idx]
        name = self.sample_name[idx]
        # subjs = self.subjs[idx]
        # print(label)

        # (C, T, V, M) -> (I, C*2, T, V, M)
        # T 帧数， V 关节点个数，M人的个数，C ：xyz坐标，这个函数目的输入6个输入
        # 绝对位置，相对位置，一阶速度，二阶速度，关节长度，关节角度
        # 所以这里的 C*2 = 6， I=3，等价于之前的C，表示关节点坐标
        data = multi_input(data, self.conn)

        return data, label, name

class UAV_Feeder3(Dataset):
    def __init__(self, phase, path, data_shape, connect_joint, debug, **kwargs):
        _, _, self.T, self.V, self.M = data_shape

        self.conn = connect_joint
        data_path = '{}/train_data.npy'.format(path)
        label_path = '{}/train_label.pkl'.format(path)

        if os.path.exists(data_path) and os.path.exists(label_path):
            self.data = np.load(data_path, mmap_mode='r')
            with open(label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')
        else:
            logging.info('')
            logging.error('Error: Do NOT exist data files: {} or {}!'.format(data_path, label_path))
            logging.info('Please generate data first!')
            raise ValueError()

        self.data = self.data[:,:,:200,:,:]

        index1 = np.load('/home/data1_4t/dpx/ResGCNv3/L1.npy')
        index2 = np.load('/home/data1_4t/dpx/ResGCNv3/L1.npy')
        index3 = np.load('/home/data1_4t/dpx/ResGCNv3/L1.npy')

        if phase=='train':

            train_index = np.concatenate((index2,index1),axis=0).tolist()
            self.data = self.data[train_index,:,:,:,:]
            self.label = [self.label[i] for i in train_index]
            self.sample_name = [self.sample_name[i] for i in train_index]

        else:

            eval_index = index3.tolist()
            self.data = self.data[eval_index,:,:,:,:]
            self.label = [self.label[i] for i in eval_index]
            self.sample_name = [self.sample_name[i] for i in eval_index]



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        label = self.label[idx]
        name = self.sample_name[idx]
        # subjs = self.subjs[idx]
        # print(label)

        # (C, T, V, M) -> (I, C*2, T, V, M)
        # T 帧数， V 关节点个数，M人的个数，C ：xyz坐标，这个函数目的输入6个输入
        # 绝对位置，相对位置，一阶速度，二阶速度，关节长度，关节角度
        # 所以这里的 C*2 = 6， I=3，等价于之前的C，表示关节点坐标
        data = multi_input(data, self.conn)

        return data, label, name