import pandas as pd
import numpy as np
from models.Utils import smiles2adjoin
import tensorflow as tf

str2num = {'<pad>': 0, 'H': 1, 'C': 2, 'O': 3, 'N': 4, 'S': 5, 'F': 6, 'Cl': 7, 'Br': 8, 'P': 9, 'I': 10, 'B': 11, 'Si': 12, 'Se': 13, '<unk>': 14, '<mask>': 15, '<global>': 16}
num2str = {i: j for j, i in str2num.items()}


class Pretrain_Dataset(object):
    """Creates masked-atom pretraining data from SMILES.
    Splits 90%/10%, applies ~15% masking, and returns (x, adj, y, weight).
    Use when training a the encoder."""
    def __init__(self, path, smiles_field='SMILES', addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.vocab = str2num
        self.devocab = num2str
        self.addH = addH

    def get_data(self):

        data = self.df
        train_idx = []
        idx = data.sample(frac=0.9).index
        train_idx.extend(idx)

        data1 = data[data.index.isin(train_idx)]
        data2 = data[~data.index.isin(train_idx)]

        self.dataset1 = tf.data.Dataset.from_tensor_slices(data1[self.smiles_field].tolist())
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([None]))).prefetch(tf.data.AUTOTUNE)

        self.dataset2 = tf.data.Dataset.from_tensor_slices(data2[self.smiles_field].tolist())
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([None]))).prefetch(tf.data.AUTOTUNE)
        return self.dataset1, self.dataset2

    def numerical_smiles(self, smiles):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        choices = np.random.permutation(len(nums_list)-1)[:max(int(len(nums_list)*0.15), 1)] + 1
        y = np.array(nums_list).astype('int64')
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = str2num['<mask>']
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 14 + 1)

        x = np.array(nums_list).astype('int64')
        weight = weight.astype('float32')
        return x, adjoin_matrix, y, weight

    def tf_numerical_smiles(self, data):
        x, adjoin_matrix, y, weight = tf.py_function(self.numerical_smiles, [data], [tf.int64, tf.float32, tf.int64, tf.float32])

        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        weight.set_shape([None])
        return x, adjoin_matrix, y, weight


class Finetune_Dataset(object):
    """Finetune data from SMILES+labels.
        Length-filter, data split (train/test/val), returns (x, adj, y)."""
    def __init__(self, path, smiles_field='SMILES', label_field='Label', max_len=240, addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path, sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.addH = addH

    def get_data(self):
        data = self.df
        lengths = [0, 60, 120, 180, 240]

        train_idx = []
        for i in range(4):
            idx = data[(data[self.smiles_field].str.len() >= lengths[i]) & (data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.8).index
            train_idx.extend(idx)

        train_data = data[data.index.isin(train_idx)]
        data = data[~data.index.isin(train_idx)]

        test_idx = []
        for i in range(4):
            idx = data[(data[self.smiles_field].str.len() >= lengths[i]) & (data[self.smiles_field].str.len() < lengths[i + 1])].sample(frac=0.5).index
            test_idx.extend(idx)

        test_data = data[data.index.isin(test_idx)]
        val_data = data[~data.index.isin(test_idx)]

        self.dataset1 = tf.data.Dataset.from_tensor_slices((train_data[self.smiles_field], train_data[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(64, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).shuffle(100).prefetch(tf.data.AUTOTUNE)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((test_data[self.smiles_field], test_data[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(tf.data.AUTOTUNE)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((val_data[self.smiles_field], val_data[self.label_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(tf.data.AUTOTUNE)

        return self.dataset1, self.dataset2, self.dataset3, test_data[self.smiles_field]

    def numerical_smiles(self, smiles, label):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)

        x = np.array(nums_list).astype('int64')
        y = np.array([label]).astype('float32')
        return x, adjoin_matrix, y

    def tf_numerical_smiles(self, smiles,label):
        x, adjoin_matrix, y = tf.py_function(self.numerical_smiles, [smiles, label], [tf.int64, tf.float32, tf.float32])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        return x, adjoin_matrix, y


class Prediction_Dataset(object):
    """Prediction dataset for raw SMILES.
        Filters by length and returns (x, adj, smiles, atom_list)."""
    def __init__(self, sml_list, max_len=240, addH=True):
        self.vocab = str2num
        self.devocab = num2str
        self.sml_list = [i for i in sml_list if len(i) < max_len]
        self.addH = addH

    def get_data(self):
        self.dataset = tf.data.Dataset.from_tensor_slices((self.sml_list,))
        self.dataset = self.dataset.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]), tf.TensorShape([None]))).cache().prefetch(tf.data.AUTOTUNE)
        return self.dataset

    def numerical_smiles(self, smiles):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles, explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list), len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)
        x = np.array(nums_list).astype('int64')
        return x, adjoin_matrix, [smiles], atoms_list

    def tf_numerical_smiles(self, smiles):
        x, adjoin_matrix, smiles, atom_list = tf.py_function(self.numerical_smiles, [smiles], [tf.int64, tf.float32, tf.string, tf.string])
        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        smiles.set_shape([1])
        atom_list.set_shape([None])
        return x, adjoin_matrix, smiles, atom_list

