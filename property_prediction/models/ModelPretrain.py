import tensorflow as tf
from models.Encoder import Encoder
from models.Utils import gelu

class ModelPretrain(tf.keras.Model):
    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=4, vocab_size=17, dropout_rate=0.1):
        super(ModelPretrain, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               input_vocab_size=vocab_size,
                               rate=dropout_rate
                               )
        self.fc1 = tf.keras.layers.Dense(d_model, activation=gelu)
        self.layernorm = tf.keras.layers.LayerNormalization(-1)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def __call__(self, x, adjoin_matrix, mask, training=False):
        x = self.encoder(x, training=training, mask=mask, adjoin_matrix=adjoin_matrix)
        x = self.fc1(x)
        x = self.layernorm(x)
        x = self.fc2(x)
        return x

