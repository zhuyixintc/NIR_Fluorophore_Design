import pandas as pd
import numpy as np
import tensorflow as tf
from rdkit import Chem

# token vocabulary
str2num = {
    '<pad>': 0,
    'H': 1,
    'C': 2,
    'N': 3,
    'O': 4,
    'F': 5,
    'S': 6,
    'Cl': 7,
    'Br': 8,
    'P': 9,
    'I': 10,
    'B': 11,
    'Si': 12,
    'Se': 13,
    '<unk>': 14,
    '<mask>': 15,
    '<global>': 16,
}

# inverse token mapping
num2str = {i: j for j, i in str2num.items()}


# convert one smiles to atoms and adjacency matrix
def smiles2graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles + ' is not valid ')
    mol = Chem.AddHs(mol)
    num_atoms = mol.GetNumAtoms()
    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())

    adjoin_matrix = np.eye(num_atoms)
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u, v] = 1.0
        adjoin_matrix[v, u] = 1.0
    return atoms_list, adjoin_matrix


class PretrainDataset(object):
    """
    Dataset class for pretraining.
    Used for building masked token prediction data from molecular-solvent SMILES.

    Main parameters:
        path: path of the input table.
        smiles_field: column name of the SMILES pair.
    """

    def __init__(self, path, smiles_field):
        self.df = pd.read_csv(path, sep='\t')
        self.smiles_field = smiles_field
        self.vocab = str2num
        self.devocab = num2str
        self.dataset1 = None
        self.dataset2 = None

    def get_data(self):
        data = self.df
        train_idx = []
        idx = data.sample(frac=0.9).index
        train_idx.extend(idx)

        data1 = data[data.index.isin(train_idx)]
        data2 = data[~data.index.isin(train_idx)]

        self.dataset1 = tf.data.Dataset.from_tensor_slices(data1[self.smiles_field].tolist())
        self.dataset2 = tf.data.Dataset.from_tensor_slices(data2[self.smiles_field].tolist())

        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(
            64,
            padded_shapes=(
                tf.TensorShape([None]),  # x
                tf.TensorShape([None, None]),  # adjoin_matrix
                tf.TensorShape([None]),  # y
                tf.TensorShape([None])  # weight
            )
        ).prefetch(tf.data.AUTOTUNE)

        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(
            64,
            padded_shapes=(
                tf.TensorShape([None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None]),
                tf.TensorShape([None])
            )
        ).prefetch(tf.data.AUTOTUNE)
        return self.dataset1, self.dataset2

    @staticmethod
    def numerical_smiles(smiles):
        smiles = smiles.numpy().decode()

        atoms_list, adjoin_matrix = smiles2graph(smiles)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]

        temp = np.ones((len(nums_list), len(nums_list)), dtype=np.float32)
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        choices = np.random.permutation(len(nums_list) - 1)[:max(int(len(nums_list) * 0.15), 1)] + 1
        y = np.array(nums_list, dtype='int64')
        weight = np.zeros(len(nums_list), dtype='float32')

        for i in choices:
            rand = np.random.rand()
            weight[i] = 1.0
            if rand < 0.8:
                nums_list[i] = str2num['<mask>']
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 14 + 1)

        x = np.array(nums_list, dtype='int64')
        return x, adjoin_matrix.astype('float32'), y, weight

    def tf_numerical_smiles(self, data):
        x, adjoin_matrix, y, weight = tf.py_function(
            self.numerical_smiles,
            [data],
            [tf.int64, tf.float32, tf.int64, tf.float32]
        )

        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        y.set_shape([None])
        weight.set_shape([None])
        return x, adjoin_matrix, y, weight


class FinetuneDataset(object):
    """
    Dataset class for finetuning.
    Used for building supervised data for downstream property prediction.

    Main parameters:
        train_path: path of the training table.
        val_path: path of the validation table.
        smiles_field: column name of the SMILES pair.
        label_field: column name of the target property.
    """

    def __init__(self, train_path, val_path, smiles_field, label_field):
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str

        self.train_df = pd.read_csv(train_path, sep='\t')
        self.val_df = pd.read_csv(val_path, sep='\t')

        def _prep(df):
            df = df.copy()
            df[label_field] = pd.to_numeric(df[label_field], errors='coerce')
            df = df.dropna(subset=[smiles_field, label_field])
            return df

        self.train_df = _prep(self.train_df)
        self.val_df = _prep(self.val_df)

        self.dataset1 = None
        self.dataset2 = None

    def get_data(self):
        train_data = self.train_df
        val_data = self.val_df

        self.dataset1 = tf.data.Dataset.from_tensor_slices((train_data[self.smiles_field], train_data[self.label_field]))
        self.dataset2 = tf.data.Dataset.from_tensor_slices((val_data[self.smiles_field], val_data[self.label_field]))

        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(
            32,
            padded_shapes=(
                tf.TensorShape([None]),  # x
                tf.TensorShape([None, None]),  # adjoin_matrix
                tf.TensorShape([1])  # y
            )
        ).shuffle(100).prefetch(tf.data.AUTOTUNE)

        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(
            32,
            padded_shapes=(
                tf.TensorShape([None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([1])
            )
        ).cache().prefetch(tf.data.AUTOTUNE)

        return self.dataset1, self.dataset2

    @staticmethod
    def numerical_smiles(smiles, label):
        smiles = smiles.numpy().decode()

        atoms_list, adjoin_matrix = smiles2graph(smiles)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]

        temp = np.ones((len(nums_list), len(nums_list)), dtype=np.float32)
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        x = np.array(nums_list, dtype='int64')
        y = np.array([label], dtype='float32')
        return x, adjoin_matrix.astype('float32'), y

    def tf_numerical_smiles(self, smiles, label):
        x, adjoin_matrix, y = tf.py_function(
            self.numerical_smiles,
            [smiles, label],
            [tf.int64, tf.float32, tf.float32]
        )
        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        y.set_shape([None])
        return x, adjoin_matrix, y


class PredictionDataset(object):
    """
    Dataset class for prediction.
    Used for converting input SMILES pairs into model-ready tensors for inference.

    Main parameters:
        sml_list: list of input SMILES pairs.
    """
    
    def __init__(self, sml_list):
        self.vocab = str2num
        self.devocab = num2str
        self.sml_list = list(sml_list)
        self.dataset = None

    def get_data(self):
        self.dataset = tf.data.Dataset.from_tensor_slices((self.sml_list,))

        self.dataset = self.dataset.map(self.tf_numerical_smiles).padded_batch(
            32,
            padded_shapes=(
                tf.TensorShape([None]),  # x
                tf.TensorShape([None, None]),  # adjoin_matrix
                tf.TensorShape([1]),  # smiles
                tf.TensorShape([None])  # atom_list
            )
        ).cache().prefetch(tf.data.AUTOTUNE)
        return self.dataset

    @staticmethod
    def numerical_smiles(smiles):
        smiles = smiles.numpy().decode()

        atoms_list, adjoin_matrix = smiles2graph(smiles)
        atoms_list = ['<global>'] + atoms_list
        nums_list = [str2num.get(i, str2num['<unk>']) for i in atoms_list]

        temp = np.ones((len(nums_list), len(nums_list)), dtype=np.float32)
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)

        x = np.array(nums_list, dtype='int64')
        return x, adjoin_matrix.astype('float32'), [smiles], atoms_list

    def tf_numerical_smiles(self, smiles):
        x, adjoin_matrix, smiles, atom_list = tf.py_function(
            self.numerical_smiles,
            [smiles],
            [tf.int64, tf.float32, tf.string, tf.string]
        )
        x.set_shape([None])
        adjoin_matrix.set_shape([None, None])
        smiles.set_shape([1])
        atom_list.set_shape([None])
        return x, adjoin_matrix, smiles, atom_list
