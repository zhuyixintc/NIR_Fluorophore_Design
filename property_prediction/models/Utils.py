import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
import numpy as np
from openbabel import openbabel as ob

def gelu(x):
    """GELU activation."""
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.)))


def scaled_dot_product_attention(q, k, v, mask, adjoin_matrix):
    """Basic attention: score, scale, mask, softmax, then mix values."""
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    """Two-layer MLP applied to each position."""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation=gelu),
        tf.keras.layers.Dense(d_model)
    ])


def obsmitosmile(smi):
    """Canonicalize a SMILES using OpenBabel."""
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = ob.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile


def smiles2adjoin(smiles, explicit_hydrogens=True, canonical_atom_order=False):
    """Parse SMILES and return (atom symbols, adjacency matrix)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        assert mol is not None, smiles + ' is not valid '

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
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
