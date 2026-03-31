import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
warnings.filterwarnings('ignore')
from core.data_processing import PretrainDataset
from core.model_transformer import ModelPretrain
from tensorflow import keras
import tensorflow as tf

# pretrain setup
keras.backend.clear_session()
pretrain_name = 'chembl_36_1M'
pretrain_path = './output/weights_pretrain'
os.makedirs(pretrain_path, exist_ok=True)
optimizer = tf.keras.optimizers.Adam(1e-4)

# model
model = ModelPretrain()

# data
train_dataset, test_dataset = PretrainDataset(path='data/{}.txt'.format(pretrain_name),
                                              smiles_field='canonical_smiles').get_data()

# input signature for tf.function
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),  # x
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),  # adjoin_matrix
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),  # y
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # char_weight
]

# metrics and loss
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# training step
@tf.function(reduce_retracing=True)
def train_step(x, adjoin_matrix, y, char_weight):
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    with tf.GradientTape() as tape:
        predictions = model([x, adjoin_matrix, mask], training=True)
        loss = loss_function(y, predictions, sample_weight=char_weight)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(loss)
    train_accuracy.update_state(y, predictions, sample_weight=char_weight)


# test step
@tf.function(input_signature=train_step_signature, reduce_retracing=True)
def test_step(x, adjoin_matrix, y, char_weight):
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    predictions = model([x, adjoin_matrix, mask], training=False)
    test_accuracy.update_state(y, predictions, sample_weight=char_weight)


# result file
file_result = './output/results_pretrain/pretrain_results.txt'
os.makedirs(os.path.dirname(file_result), exist_ok=True)
result = 'Epoch\tStep\tTrain_Loss\tTrain_Accuracy\tTest_Accuracy'
with open(file_result, 'w') as f:
    f.write(result + '\n')
print(result)

# training loop
global_step = 0
for epoch in range(10):
    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (x, adjoin_matrix, y, char_weight)) in enumerate(train_dataset):
        train_step(x, adjoin_matrix, y, char_weight)
        global_step += 1

        # evaluate at the first step and every 2000 steps
        if (global_step == 1) or (global_step % 2000 == 0):
            f_epoch = epoch + 1
            f_batch = global_step
            f_train_loss = tf.get_static_value(train_loss.result())
            f_train_accuracy = tf.get_static_value(train_accuracy.result())

            for x_t, adjoin_matrix_t, y_t, char_weight_t in test_dataset:
                test_step(x_t, adjoin_matrix_t, y_t, char_weight_t)
            f_test_accuracy = tf.get_static_value(test_accuracy.result())

            # save log
            result = '\t'.join(map(str, [f_epoch, f_batch, f_train_loss, f_train_accuracy, f_test_accuracy]))
            with open(file_result, 'a') as f:
                f.write(result + '\n')

            print(f"{f_epoch}\t{f_batch}\t{float(f_train_loss):.4f}\t{float(f_train_accuracy):.4f}\t{float(f_test_accuracy):.4f}")

            # reset running metrics after logging
            test_accuracy.reset_states()
            train_loss.reset_states()
            train_accuracy.reset_states()

    # save weights after each epoc
    model.save_weights(f"{pretrain_path}/{pretrain_name}_{epoch + 1}.weights.h5")

