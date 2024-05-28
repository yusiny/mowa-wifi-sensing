import os
import random
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from runner.utils import get_config, torch_seed
from scipy.signal import savgol_filter  # 이동 평균 필터 대신 Savitzky-Golay 필터를 사용

torch_seed(40)

# CSI dataset loader for Meta-Learning
class CSIDataset(data.Dataset):
    def __init__(self, data_path, num_support, num_query, win_size=10, mode='train', amp=True):
        self.mode = mode
        self.win_size = win_size
        self.config = get_config('config.yaml')

        if self.mode == "train" :
            self.labels = self.config['FSL']['dataset']['train_activity_labels']

        elif self.mode == "test" : 
            self.labels = self.config['FSL']['dataset']['test_activity_labels']

        self.amp = amp

        # Read CSI file and convert to dataframe
        self.data_df = dict()
        self.min_num_wd = int(1e9)  # for balanced dataset class

        for atv in self.labels:
            f_path = os.path.join(data_path, atv + '.csv')
            self.data_df[atv] = pd.read_csv(f_path)

        # Generated window size CSI data
        self.data = self.generate_windows()

        # Variables for episodic learning
        self.num_support = num_support
        self.num_query = num_query
        self.num_episode = self.min_num_wd // (self.num_support + self.num_query)

    def generate_windows(self):
        win_dict = dict()
        for atv in self.data_df.keys():
            windows = list()
            df = self.data_df[atv]
            
            num_win = len(df) // self.win_size

            self.min_num_wd = min(self.min_num_wd, num_win)

            for i in range(num_win):
                wd = df.iloc[i*self.win_size:i*self.win_size + self.win_size, 2:]
                wd = wd.astype(complex)
            
                if self.amp is True:
                    wd = wd.apply(lambda x: x.abs())
                
                wd = wd.to_numpy()
                windows.append(wd)
            
            win_dict[atv] = np.array(windows)
        return win_dict

    def __len__(self):
        return self.num_episode

    def __getitem__(self, idx):

        # idx means index of episode
        sample = dict()

        for label in self.labels:
            sample[label] = dict()

            # support set
            support_indices = random.sample(range(self.min_num_wd), self.num_support)
            support_set = [self.data[label][i] for i in support_indices]
            sample[label]['support'] = np.array(support_set)

            # query set
            query_indices = list(set(range(self.min_num_wd)) - set(support_indices))
            query_indices = random.sample(query_indices, self.num_query)
            query_set = [self.data[label][i] for i in query_indices]
            sample[label]['query'] = np.array(query_set)

        return sample
    

# Supervised learning dataset
class SVLDataset(data.Dataset):
    def __init__(self, data_path, win_size=10, mode='train', train_proportion=0.8, amp=True, filter_window=5, filter_polyorder=2):
        self.mode = mode
        self.win_size = win_size
        self.config = get_config('config.yaml')
        self.labels = self.config['SVL']['dataset']['activity_labels']
        self.amp = amp
        self.train_proportion = train_proportion
        
        self.filter_window = filter_window  # 필터 윈도우 크기
        self.filter_polyorder = filter_polyorder  # 필터 다항식 차수

        # Read CSI file and convert to dataframe
        self.data_df = dict()
        self.min_data_len = int(1e9)

        for atv in self.labels:
            # print(data_path, atv + '.csv')
            f_path = os.path.join(data_path, atv + '.csv')
            self.data_df[atv] = pd.read_csv(f_path)
            self.min_data_len = min(self.min_data_len, len(self.data_df[atv]))


        # Generated window size CSI data
        self.min_num_wd = self.min_data_len // self.win_size  # for balanced dataset class
        
        self.num_train = int(self.min_num_wd * self.train_proportion)
        self.num_test = self.min_num_wd - self.num_train
        
        self.data_x, self.data_y = self.generate_windows()

    def generate_windows(self):
        win_data_x = list()
        data_y = list()
        atvs = list()
        atvs = list(self.data_df.keys())

        # 모든 데이터프레임의 최대 열 수 계산
        max_cols = max([self.data_df[atv].shape[1] for atv in atvs]) - 2  # 첫 두 열은 제외
        
        for idx, atv in enumerate(atvs):
            windows = list()
            df = self.data_df[atv]

            y_label = idx
            
            # Apply Savitzky-Golay filter to the real and imaginary parts separately
            def filter_complex_series(series):
                real_filtered = savgol_filter(series.apply(lambda z: np.real(complex(z))), self.filter_window, self.filter_polyorder)
                imag_filtered = savgol_filter(series.apply(lambda z: np.imag(complex(z))), self.filter_window, self.filter_polyorder)
                return real_filtered + 1j * imag_filtered
            
            for col in df.columns[2:]:
                df[col] = filter_complex_series(df[col])

            for i in range(self.min_num_wd):
                wd = df.iloc[i*self.win_size:i*self.win_size + self.win_size, 2:]
                wd = wd.astype(complex)
            
                if self.amp is True:
                    wd = wd.apply(lambda x: x.abs())
                
                wd = wd.to_numpy()
                
                if wd.shape[1] < max_cols:
                    # 부족한 열을 mean 값으로 채움
                    diff = max_cols - wd.shape[1]
                    mean_values = wd.mean(axis=1)
                    padding = np.tile(mean_values[:, None], (1, diff))
                    wd = np.hstack((wd, padding))
                    
                windows.append(wd)
            
            if self.mode == 'train':
                windows = windows[:self.num_train]
                data_y.extend([y_label for _ in range(len(windows))])
            elif self.mode == 'test':
                windows = windows[self.num_train:]
                data_y.extend([y_label for _ in range(len(windows))])
            
            win_data_x.extend(windows)
            
        return np.array(win_data_x), np.array(data_y)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]

# Few learning dataset
class FSLDataset(data.Dataset) :
    def __init__(self, data_path, win_size=10, mode='train', amp=True, mac=True, time=True):
        self.mode = mode
        self.win_size = win_size
        self.config = get_config('config.yaml')
        
        if self.mode == "train" :
            self.labels = self.config['FSL']['dataset']['train_activity_labels']

        elif self.mode == "test" : 
            self.labels = self.config['FSL']['dataset']['test_activity_labels']       
             
        self.amp = amp
        self.mac = mac
        self.time = time

        # Read CSI file and convert to dataframe
        self.data_df = dict()
        self.min_data_len = int(1e9)

        for atv in self.labels:
            f_path = os.path.join(data_path, atv + '.csv')
            self.data_df[atv] = pd.read_csv(f_path)
            self.min_data_len = min(self.min_data_len, len(self.data_df[atv]))

        # Generated window size CSI data
        self.min_num_wd = self.min_data_len // self.win_size  # for balanced dataset class

        self.data_x, self.data_y = self.generate_windows()

    def generate_windows(self) :
        win_data_x = list()
        data_y = list()
        atvs = list()

        atvs = list(self.data_df.keys())

        for idx, atv in enumerate(atvs):
            windows = list()
            df = self.data_df[atv]

            y_label = idx

            for i in range(self.min_num_wd):
                if self.mac == True and self.time == True :
                    wd = df.iloc[i*self.win_size:i*self.win_size + self.win_size, 2:]
                elif self.mac == False and self.time == False :
                    wd = df.iloc[i*self.win_size:i*self.win_size + self.win_size, 0:]
                wd = wd.astype(complex)
            
                if self.amp is True:
                    wd = wd.apply(lambda x: x.abs())
                
                wd = wd.to_numpy()
                windows.append(wd)
            win_data_x.extend(windows)
            data_y.extend([y_label for _ in range(len(windows))])

        return np.array(win_data_x), np.array(data_y)

    def __len__(self):
            return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]



if __name__ == '__main__':
    data = CSIDataset('../csi_dataset/domain_B',5,10,win_size=10,mode='train')
    print(data.__len__())
    print(data.__getitem__(0)['empty']['support'].shape)