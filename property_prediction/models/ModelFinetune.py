import tensorflow as tf
from models.Encoder import Encoder


class ModelFinetune(tf.keras.Model):
    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=4, vocab_size=17, dropout_rate=0.1,
                 dense_dropout=0.1):
        super(ModelFinetune, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               input_vocab_size=vocab_size,
                               rate=dropout_rate
                               )

        self.fc1 = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(0.1))
        self.dropout = tf.keras.layers.Dropout(dense_dropout)
        self.fc2 = tf.keras.layers.Dense(1)

    def __call__(self, x, adjoin_matrix, mask, training=False):
        x = self.encoder(x, training=training, mask=mask, adjoin_matrix=adjoin_matrix)
        x = x[:, 0, :]
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return x

