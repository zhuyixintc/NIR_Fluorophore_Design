import multiprocessing
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
warnings.filterwarnings('ignore')
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from models.DataProcessing import Finetune_Dataset
from sklearn.metrics import r2_score
from models.ModelFinetune import ModelFinetune
from models.ModelPretrain import ModelPretrain
from sklearn.metrics import mean_absolute_error


def main(seed, task):
    keras.backend.clear_session()

    pretrain_name = 'chembl_1M'
    pretrain_path = './weights_pretrain'

    finetune_name = task
    finetune_path = './weights_finetune'

    pretraining = True

    trained_epoch = 10

    file_result_lc = './results_finetune/learning_curve_' + str(finetune_name) + '_' + str(seed) + '.txt'
    with open(file_result_lc, 'w') as f:
        f.write('Epoch\tTrain_MSE\tVal_MSE' + '\n')

    val_mse, val_rmse, val_mae, val_r2 = None, None, None, None

    print('_' * 40 + finetune_name + '_' + str(seed) + '_' + 'Starting!' + '_' * 40)

    seed = seed

    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_dataset, test_dataset, val_dataset, test_smi = Finetune_Dataset('data/{}.txt'.format(finetune_name)).get_data()

    x, adjoin_matrix, y = next(iter(train_dataset.take(1)))

    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = ModelFinetune()

    if pretraining:
        temp = ModelPretrain()
        pred = temp(x, adjoin_matrix=adjoin_matrix, mask=mask, training=True)
        temp.load_weights(pretrain_path+'/{}_{}'.format(pretrain_name, trained_epoch))
        temp.encoder.save_weights(pretrain_path+'/_weights_encoder_{}_{}'.format(pretrain_name, trained_epoch), save_format='tf')
        del temp

        pred = model(x, adjoin_matrix=adjoin_matrix, mask=mask, training=True)
        model.encoder.load_weights(pretrain_path+'/_weights_encoder_{}_{}'.format(pretrain_name, trained_epoch))
        print('load_weights')

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    best_val = float('-inf')
    stopping_monitor = 0
    for epoch in range(150):
        mse_object = tf.keras.metrics.MeanSquaredError()
        for x, adjoin_matrix, y in train_dataset:
            with tf.GradientTape() as tape:
                seq = tf.cast(tf.math.equal(x, 0), tf.float32)
                mask = seq[:, tf.newaxis, tf.newaxis, :]
                preds = model(x, adjoin_matrix=adjoin_matrix, mask=mask, training=True)
                loss = tf.reduce_mean(tf.square(y-preds))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                mse_object.update_state(y, preds)

        y_true = []
        y_preds = []
        for x, adjoin_matrix, y in val_dataset:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x, adjoin_matrix=adjoin_matrix, mask=mask, training=False)
            y_true.append(y.numpy())
            y_preds.append(preds.numpy())
        y_true = np.concatenate(y_true, axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds, axis=0).reshape(-1)

        # evaluate val
        val_mse = keras.metrics.MSE(y_true, y_preds).numpy()
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(y_true, y_preds)
        val_r2 = r2_score(y_true, y_preds)

        print('epoch: ', epoch,
              'loss: {:.4f}'.format(loss.numpy().item()),
              'train mse: {:.4f}'.format(mse_object.result().numpy().item()),
              'val mse:{:.4f}'.format(val_mse)
              )

        with open(file_result_lc, 'a') as f:
            f.write(str(epoch) + '\t' + str(mse_object.result().numpy().item()) + '\t' + str(val_mse) + '\n')

        if val_r2 > best_val:
            best_val = val_r2
            stopping_monitor = 0
            model.save_weights(finetune_path + '/{}_{}'.format(finetune_name, seed), save_format='tf')
        else:
            stopping_monitor += 1
        print('best r2: {:.4f}'.format(best_val))
        if stopping_monitor > 0:
            print('stopping_monitor:', stopping_monitor)
        if stopping_monitor > 15:
            break

    data_saving_y_true = []
    data_saving_y_predict = []
    data_saving_test_smi = []
    y_true = []
    y_preds = []

    model.load_weights(finetune_path + '/{}_{}'.format(finetune_name, seed))
    for x, adjoin_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, adjoin_matrix=adjoin_matrix, mask=mask, training=False)
        y_true.append(y.numpy())
        y_preds.append(preds.numpy())
    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    for i in y_true:
        data_saving_y_true.append({'y_true': i})
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
    for i in y_preds:
        data_saving_y_predict.append({'y_predict': i})
    for i in test_smi.tolist():
        data_saving_test_smi.append({'SMILES': i})
    data_saving_y_true = pd.DataFrame(data_saving_y_true)
    data_saving_y_predict = pd.DataFrame(data_saving_y_predict)
    data_saving_test_smi = pd.DataFrame(data_saving_test_smi)
    data_saving = pd.concat([data_saving_test_smi, data_saving_y_true, data_saving_y_predict], axis=1)
    data_saving.to_csv('./results_finetune/' + str(finetune_name) + '_' + str(seed) + '.csv', index=False, header=True)

    test_r2 = r2_score(y_true, y_preds)
    test_mse = keras.metrics.MSE(y_true.reshape(-1), y_preds.reshape(-1)).numpy()
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_true, y_preds)

    print('test mse:{:.4f}'.format(test_mse), 'test r2:{:.4f}'.format(test_r2))

    return val_mse, val_rmse, val_mae, val_r2, test_mse, test_rmse, test_mae, test_r2


def run(task):
    file_result_pfm_val = './results_finetune/model_performance_' + str(task) + '_val.txt'
    with open(file_result_pfm_val, 'a') as f:
        f.write('Seed\tMSE\tRMSE\tMAE\tR2' + '\n')

    file_result_pfm_test = './results_finetune/model_performance_' + str(task) + '_test.txt'
    with open(file_result_pfm_test, 'a') as f:
        f.write('Seed\tMSE\tRMSE\tMAE\tR2' + '\n')

    for i in [17, 72, 97, 8, 32, 15, 63, 57, 60, 83]:
        val_mse, val_rmse, val_mae, val_r2, test_mse, test_rmse, test_mae, test_r2 = main(i, task)
        with open(file_result_pfm_val, 'a') as f:
            f.write(str(i) + '\t' + str(val_mse) + '\t' + str(val_rmse) + '\t' + str(val_mae) + '\t' + str(val_r2) + '\n')
        with open(file_result_pfm_test, 'a') as f:
            f.write(str(i) + '\t' + str(test_mse) + '\t' + str(test_rmse) + '\t' + str(test_mae) + '\t' + str(test_r2) + '\n')


if __name__ == "__main__":
    for j in ['abs', 'ex', 'plqy']:
        job = multiprocessing.Process(target=run, args=(j,))
        job.start()
        job.join()
        job.close()

