import pandas as pd
import sascorer
from rdkit import Chem
from tqdm import tqdm


# solvent list
solvent_list = ['ClC(Cl)Cl',
                'ClCCl',
                'CS(C)=O',
                'CCO',
                'CCCCCC',
                'CC#N',
                'CO',
                'C1CCOC1',
                'Cc1ccccc1',
                'O']

# combine smiles
df = pd.read_csv('./data/generated_molecules.csv')
df = df.drop_duplicates()
data = []
es = 0

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    i = row['SMILES']
    mol = Chem.MolFromSmiles(i)
    sa_score = sascorer.calculateScore(mol)
    if sa_score < 6:
        es += 1
    for j in range(10):
        data.append({'A': row['A'], 'D1': row['D1'], 'D2': row['D2'],
                     'SMILES': i + '.' + solvent_list[j], 'SA_Score': sa_score})

data = pd.DataFrame(data)
data.to_csv('./data/generated_molecules_in_solvents.csv', index=False, header=True)
print(es)
print('Total number of generated molecules:', len(df))  # 20949
print('Percentage of ES molecules:', es / len(df))  # 1.0

