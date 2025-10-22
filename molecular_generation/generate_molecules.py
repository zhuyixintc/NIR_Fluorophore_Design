import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def m2s(m):
    smiles = Chem.MolToSmiles(m)
    return smiles


def s2m(smi):
    m = Chem.MolFromSmiles(smi)
    return m


# Load acceptor and donor units.
dfa = pd.read_csv('your_acceptor_path')
dfd = pd.read_csv('your_donor_path')


def remove_dp(smi):
    mol = s2m(smi)
    prod = Chem.RemoveHs(AllChem.ReplaceSubstructs(mol, s2m('*'), s2m('[H]'), True)[0])
    back_smi = m2s(prod)
    return back_smi


def gen(a, d):
    gen_list = []
    x1 = s2m(a)
    x2 = s2m(d)
    combo = Chem.CombineMols(x1, x2)
    atom1 = []

    for atom in x1.GetAtoms():
        atom1.append(
            [atom.GetIdx(), atom.GetSymbol(),
             [(nbr.GetIdx(), nbr.GetSymbol()) for nbr in atom.GetNeighbors()]])
    atom1_symbol = [elt[1] for elt in atom1]
    idx1 = [o for o, p in list(enumerate(atom1_symbol)) if p == '*']
    atom_combo = []

    for atom in combo.GetAtoms():
        atom_combo.append(
            [atom.GetIdx(), atom.GetSymbol(),
             [(nbr.GetIdx(), nbr.GetSymbol()) for nbr in atom.GetNeighbors()]])
    atom_combo_symbol = [elt[1] for elt in atom_combo]
    idx_combo = [o for o, p in list(enumerate(atom_combo_symbol)) if p == '*']
    idx2 = idx_combo[len(idx1):]

    for i in range(len(idx1)):
        for j in range(len(idx2)):
            edcombo = Chem.EditableMol(combo)

            x = idx1[i]
            y = idx2[j]
            p1 = atom_combo[x][2][0][0]
            p2 = atom_combo[y][2][0][0]
            edcombo.AddBond(p1, p2, order=Chem.rdchem.BondType.SINGLE)
            atom_to_remove = [x] + idx2
            atom_to_remove.sort(reverse=True)

            for atom in atom_to_remove:
                edcombo.RemoveAtom(atom)
            back = edcombo.GetMol()
            back_smi = Chem.MolToSmiles(back)
            gen_list.append(back_smi)
    gen_list = list(dict.fromkeys(gen_list))
    return gen_list


dfm = []
for a in dfa:
    for d in dfd:
        for m in gen(a, d):
            dfm.append({'A': remove_dp(a), 'D1': remove_dp(d), 'SMILES': m})
dfm = pd.DataFrame(dfm)
dfm = dfm.drop_duplicates(subset='SMILES', keep='first')

dfp = []
for index, row in dfm.iterrows():
    for d in dfd:
        for p in gen(row['SMILES'], d):
            if '*' in p:
                dfp.append({'A': row['A'], 'D1': row['D1'], 'D2': remove_dp(d), 'SMILES': remove_dp(p)})
            else:
                dfp.append({'A': row['A'], 'D1': row['D1'], 'D2': remove_dp(d), 'SMILES': p})
dfp = pd.DataFrame(dfp)
dfp = dfp.drop_duplicates(subset='SMILES', keep='first')
print('form D-A-D structures done')

