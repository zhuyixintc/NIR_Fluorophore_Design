from tensorflow import keras
keras.backend.clear_session()
import tensorflow as tf
from models.ModelPretrain import ModelPretrain
from models.DataProcessing import Pretrain_Dataset
import time


pretrain_name = 'your_pretrain_database'
pretrain_path = 'your_pretrain_weights_path'

optimizer = tf.keras.optimizers.Adam(1e-4)

model = ModelPretrain()

train_dataset, test_dataset = Pretrain_Dataset(path='data/{}.txt'.format(pretrain_name)).get_data()

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
]

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


@tf.function(reduce_retracing=True)
def train_step(x, adjoin_matrix, y, char_weight):
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    with tf.GradientTape() as tape:
        predictions = model(x, adjoin_matrix=adjoin_matrix, mask=mask, training=True)
        loss = loss_function(y, predictions, sample_weight=char_weight)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(loss)
    train_accuracy.update_state(y, predictions, sample_weight=char_weight)


@tf.function(input_signature=train_step_signature, reduce_retracing=True)
def test_step(x, adjoin_matrix, y, char_weight):
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    predictions = model(x, adjoin_matrix=adjoin_matrix, mask=mask, training=False)
    test_accuracy.update_state(y, predictions, sample_weight=char_weight)


file_result = 'your_path_to_save_results'
result = 'Epoch\tBatch\tTrain_Loss\tTrain_Accuracy\tTest_Accuracy'
with open(file_result, 'w') as f:
    f.write(result + '\n')
print(result)
template = "{:3}{:8}{:13.4f}{:13.4f}{:15.4f}"

for epoch in range(10):
    start = time.time()
    train_loss.reset_states()

    for (batch, (x, adjoin_matrix, y, char_weight)) in enumerate(train_dataset):
        train_step(x, adjoin_matrix, y, char_weight)

        if batch % 1000 == 0:
            f_epoch = epoch + 1
            f_batch = batch
            f_train_loss = tf.get_static_value(train_loss.result())
            f_train_accuracy = tf.get_static_value(train_accuracy.result())

            for x_t, adjoin_matrix_t, y_t, char_weight_t in test_dataset:
                test_step(x_t, adjoin_matrix_t, y_t, char_weight_t)
            f_test_accuracy = tf.get_static_value(test_accuracy.result())

            result = '\t'.join(map(str, [f_epoch, f_batch, f_train_loss, f_train_accuracy, f_test_accuracy]))
            with open(file_result, 'a') as f:
                f.write(result + '\n')

            print(template.format(f_epoch, f_batch, f_train_loss, f_train_accuracy, f_test_accuracy))

            test_accuracy.reset_states()
            train_accuracy.reset_states()

    model.save_weights(pretrain_path + '/{}_{}'.format(pretrain_name, epoch+1), save_format='tf')

