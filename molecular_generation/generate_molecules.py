import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdDepictor
from rdkit import RDLogger

rdDepictor.SetPreferCoordGen(True)
RDLogger.DisableLog('rdApp.*')


def m2s(m):
    smiles = Chem.MolToSmiles(m)
    return smiles


def s2m(smi):
    m = Chem.MolFromSmiles(smi)
    return m


# data loading
dfa = pd.read_csv('./data/acceptor_raw.csv')
dfd = pd.read_csv('./data/donor_raw.csv')


def remove_dp(smi):
    mol = s2m(smi)
    prod = Chem.RemoveHs(AllChem.ReplaceSubstructs(mol, s2m('*'), s2m('[H]'), True)[0])
    back_smi = m2s(prod)
    return back_smi


# add structures
def gen(a, d):
    gen_list = []
    x1 = s2m(a)
    x2 = s2m(d)
    combo = Chem.CombineMols(x1, x2)

    # get main atoms
    atom1 = []
    for atom in x1.GetAtoms():
        atom1.append(
            [atom.GetIdx(), atom.GetSymbol(),
             [(nbr.GetIdx(), nbr.GetSymbol()) for nbr in atom.GetNeighbors()]])
    atom1_symbol = [elt[1] for elt in atom1]
    idx1 = [o for o, p in list(enumerate(atom1_symbol)) if p == '*']

    # get combo atoms
    atom_combo = []
    for atom in combo.GetAtoms():
        atom_combo.append(
            [atom.GetIdx(), atom.GetSymbol(),
             [(nbr.GetIdx(), nbr.GetSymbol()) for nbr in atom.GetNeighbors()]])
    atom_combo_symbol = [elt[1] for elt in atom_combo]
    idx_combo = [o for o, p in list(enumerate(atom_combo_symbol)) if p == '*']
    idx2 = idx_combo[len(idx1):]

    # generate molecules
    for i in range(len(idx1)):
        for j in range(len(idx2)):
            edcombo = Chem.EditableMol(combo)

            # get linking index and add single bond
            x = idx1[i]
            y = idx2[j]
            p1 = atom_combo[x][2][0][0]
            p2 = atom_combo[y][2][0][0]
            edcombo.AddBond(p1, p2, order=Chem.rdchem.BondType.SINGLE)

            # remove predefined linking points (should start from the highest order)
            atom_to_remove = [x] + idx2
            atom_to_remove.sort(reverse=True)
            for atom in atom_to_remove:
                edcombo.RemoveAtom(atom)

            # back to mol
            back = edcombo.GetMol()
            back_smi = Chem.MolToSmiles(back)
            gen_list.append(back_smi)
    gen_list = list(dict.fromkeys(gen_list))
    return gen_list


# molecular generation
# D-A-D structures
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

df1 = []
for index, row in dfp.iterrows():
    smi = row['SMILES']
    check_smi = m2s(s2m(smi)) + '.ClC(Cl)Cl'
    if smi is not None and len(check_smi) < 240:
        smiles = m2s(s2m(smi))
        df1.append({'A': row['A'], 'D1': row['D1'], 'D2': row['D2'], 'SMILES': smiles})
    else:
        pass
df1 = pd.DataFrame(df1)
df1 = df1.drop_duplicates(subset='SMILES', keep='first')

# remove molecules in data set
dfA = pd.read_csv('../property_prediction/data/abs.txt', sep='\t')['SMILES'].tolist()
dfB = pd.read_csv('../property_prediction/data/ex.txt', sep='\t')['SMILES'].tolist()
dfC = pd.read_csv('../property_prediction/data/plqy.txt', sep='\t')['SMILES'].tolist()
df2 = dfA + dfB + dfC
df2 = list(dict.fromkeys(df2))
database = []
for i in df2:
    database.append(i.split('.')[0])

df3 = []
for index, row in df1.iterrows():
    if row['SMILES'] in database:
        pass
    else:
        df3.append({'A': row['A'], 'D1': row['D1'], 'D2': row['D2'], 'SMILES': row['SMILES']})

df3 = pd.DataFrame(df3)
df3 = df3.drop_duplicates(subset='SMILES', keep='first')
df3.to_csv('./data/generated_molecules.csv', index=False, header=True)

