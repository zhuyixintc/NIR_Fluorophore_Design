import tensorflow as tf
from models.EncoderLayer import EncoderLayer


class Encoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def __call__(self, x, training, mask, adjoin_matrix):
        adjoin_matrix = adjoin_matrix[:, tf.newaxis, :, :]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attention_weights = self.enc_layers[i](x, training, mask, adjoin_matrix)
        return x

