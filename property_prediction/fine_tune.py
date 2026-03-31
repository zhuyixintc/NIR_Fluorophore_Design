import multiprocessing
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
warnings.filterwarnings('ignore')
from core.data_processing import FinetuneDataset, PredictionDataset
from core.model_transformer import ModelFinetune, ModelPretrain
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm

# parameters
DATA_DIR = './data/splits'
TASKS = ['abs', 'em', 'plqy']
TRAIN_SETS = ['chemfluor']
#TASKS = ['abs', 'em', 'plqy']
#TRAIN_SETS = ['chemfluor', 'deep4chem', 'chemfluor_expanded']
N_SEEDS = 3
N_EPOCH = 200


######################
# train + val + test #
######################
def main(seed, task, train_set):
    keras.backend.clear_session()

    # pretrain setup
    pretrain_name = 'chembl_36_1M'
    pretrain_path = './output/weights_pretrain'

    # finetune setup
    finetune_name = f'{task}_train_on_{train_set}'
    finetune_path = './output/weights_finetune'
    result_path = './output/results_finetune'
    os.makedirs(finetune_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    # load pretrained encoder weights
    pretraining = True
    trained_epoch = 10

    # random seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # read train and val data
    train_path = f'{DATA_DIR}/{train_set}_{task}_train.txt'
    val_path = f'{DATA_DIR}/{train_set}_{task}_val.txt'
    ds = FinetuneDataset(train_path, val_path,
                         smiles_field='mol_solvent_smiles',
                         label_field=task)
    train_dataset, val_dataset = ds.get_data()

    # build one batch to initialize model
    x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = ModelFinetune()

    # transfer encoder weights from pretraining
    if pretraining:
        temp = ModelPretrain()
        _ = temp([x, adjoin_matrix, mask], training=False)
        temp.load_weights(f"{pretrain_path}/{pretrain_name}_{trained_epoch}.weights.h5")

        _ = model([x, adjoin_matrix, mask], training=False)
        model.encoder.set_weights(temp.encoder.get_weights())
        del temp

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # training step
    @tf.function(reduce_retracing=True)
    def train_step(x, adjoin_matrix, y):
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        with tf.GradientTape() as tape:
            preds = model([x, adjoin_matrix, mask], training=True)
            loss = tf.reduce_mean(tf.square(y - preds))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return preds

    #########
    # train #
    #########
    lc_lines = []
    best_val = float('inf')
    stopping_monitor = 0
    last_postfix = {}

    pbar = tqdm(total=N_EPOCH, desc=f'{finetune_name}|Seed={seed}', leave=True)
    for epoch in range(N_EPOCH):
        mse_object = tf.keras.metrics.MeanSquaredError()

        # train on all batches
        for x, adjoin_matrix, y in train_dataset:
            preds = train_step(x, adjoin_matrix, y)
            mse_object.update_state(y, preds)

        #######
        # val #
        #######
        y_true = []
        y_preds = []

        # run validation
        for x, adjoin_matrix, y in val_dataset:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model([x, adjoin_matrix, mask], training=False)
            y_true.append(y)
            y_preds.append(preds)
        y_true = tf.reshape(tf.concat(y_true, axis=0), [-1]).numpy()
        y_preds = tf.reshape(tf.concat(y_preds, axis=0), [-1]).numpy()
        val_mse = np.mean((y_true - y_preds) ** 2)
        val_r2 = r2_score(y_true, y_preds)

        # save learning curve info
        lc_lines.append({'Epoch': int(epoch), 'Train_MSE': float(mse_object.result().numpy().item()), 'Val_MSE': float(val_mse), 'Val_R2': float(val_r2)})

        # save best model by val mse
        if val_mse < best_val:
            best_val = val_mse
            stopping_monitor = 0
            model.save_weights(f"{finetune_path}/{finetune_name}_{seed}.weights.h5")
        else:
            stopping_monitor += 1

        # update progress bar
        last_postfix = {'Train_MSE': f'{mse_object.result().numpy().item():.4f}',
                        'Val_MSE': f'{val_mse:.4f}',
                        'Val_R2': f'{val_r2:.4f}',
                        'Best_MSE': f'{best_val:.4f}',
                        'Stop_Monitor': stopping_monitor}
        pbar.set_postfix(last_postfix)
        pbar.update(1)

        # early stopping
        if stopping_monitor >= 20:
            break

    # save learning curve
    lc_lines = pd.DataFrame(lc_lines)
    lc_lines.to_csv(f'./output/results_finetune/learning_curve_{finetune_name}_{seed}.txt', sep="\t", index=False, header=True)

    ########
    # test #
    ########
    # load model
    model.load_weights(f"{finetune_path}/{finetune_name}_{seed}.weights.h5")

    # read test data
    test_set = train_set
    test_path = f"{DATA_DIR}/{test_set}_{task}_test.txt"
    test_df = pd.read_csv(test_path, sep='\t').copy()
    test_df[task] = pd.to_numeric(test_df[task], errors='coerce')
    test_df = test_df.dropna(subset=['mol_solvent_smiles', task])
    test_smi = test_df['mol_solvent_smiles'].astype(str).tolist()
    y_true = test_df[task].to_numpy(np.float32)
    ds = PredictionDataset(test_smi)
    test_dataset = ds.get_data()

    # run prediction
    y_preds = []
    for x, adjoin_matrix, _, _ in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model([x, adjoin_matrix, mask], training=False)
        y_preds.append(preds)
    y_preds = tf.reshape(tf.concat(y_preds, axis=0), [-1]).numpy()

    # save test predictions
    data_saving = pd.DataFrame({'mol_solvent_smiles': test_smi, 'y_true': y_true, 'y_predict': y_preds})
    data_saving.to_csv(f'./output/results_finetune/predictions_{finetune_name}_test_on_{test_set}_{seed}.csv', index=False, header=True)

    # calculate and save test metrics
    test_mse = float(np.mean((y_true - y_preds) ** 2))
    test_rmse = float(np.sqrt(test_mse))
    test_mae = float(mean_absolute_error(y_true, y_preds))
    test_r2 = float(r2_score(y_true, y_preds))
    test_lines = pd.DataFrame([{'Seed': int(seed), 'MSE': float(test_mse), 'RMSE': float(test_rmse), 'MAE': float(test_mae), 'R2': float(test_r2)}])
    test_lines.to_csv(f'./output/results_finetune/metrics_{finetune_name}_test_on_{test_set}_{seed}.txt', sep='\t', index=False)

    # update progress bar with test result
    last_postfix['Test_R2'] = f'{test_r2:.4f}'
    pbar.set_postfix(last_postfix)
    pbar.refresh()
    pbar.close()


if __name__ == '__main__':
    for task in TASKS:
        for train_set in TRAIN_SETS:
            for seed in range(1, N_SEEDS + 1):
                job = multiprocessing.Process(target=main, args=(seed, task, train_set))
                job.start()
                job.join()
                job.close()


'''

'''
