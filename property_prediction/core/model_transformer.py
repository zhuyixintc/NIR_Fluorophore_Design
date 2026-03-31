import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head self-attention layer.
    Used for learning interactions between token representations.

    Main parameters:
        d_model: hidden dimension of token representations.
        num_heads: number of attention heads.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    @staticmethod
    def scaled_dot_product_attention(q, k, v, attn_mask, adjoin_matrix):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if attn_mask is not None:
            scaled_attention_logits += (attn_mask * -1e9)
        if adjoin_matrix is not None:
            scaled_attention_logits += adjoin_matrix
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def call(self, inputs, training=None, mask=None):
        q, k, v, adjoin_matrix, attn_mask = inputs
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, attn_mask, adjoin_matrix)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    """
    Transformer encoder block with self-attention and feed-forward network.
    Used for updating token representations in one encoder block.

    Main parameters:
        d_model: hidden dimension of token representations.
        num_heads: number of attention heads.
        dff: hidden dimension of the feed-forward network.
        rate: dropout rate.
    """

    def __init__(self, d_model, num_heads, dff, rate):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(dff, activation=tf.keras.activations.gelu), tf.keras.layers.Dense(d_model)])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None, mask=None):
        x, adjoin_matrix, attn_mask = inputs
        attn_output, attention_weights = self.mha([x, x, x, adjoin_matrix, attn_mask], training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2, attention_weights


class Encoder(tf.keras.layers.Layer):
    """
    Transformer encoder with token embedding and stacked encoder blocks.
    Used for encoding input tokens into contextual representations.

    Main parameters:
        num_layers: number of encoder blocks.
        d_model: hidden dimension of token representations.
        num_heads: number of attention heads.
        dff: hidden dimension of the feed-forward network.
        input_vocab_size: vocabulary size of input tokens.
        rate: dropout rate.
    """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=None, mask=None):
        x, adjoin_matrix, attn_mask = inputs
        adjoin_matrix = adjoin_matrix[:, tf.newaxis, :, :]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.dropout(x, training=training)

        for enc_layer in self.enc_layers:
            x, attention_weights = enc_layer([x, adjoin_matrix, attn_mask], training=training)
        return x


class ModelPretrain(tf.keras.Model):
    """
    Pretraining model for masked token prediction.
    Used for learning general token representations before finetuning.

    Main parameters:
        num_layers: number of encoder blocks.
        d_model: hidden dimension of token representations.
        dff: hidden dimension of the feed-forward network.
        num_heads: number of attention heads.
        vocab_size: vocabulary size of prediction targets.
        dropout_rate: dropout rate.
    """

    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=4, vocab_size=17, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               input_vocab_size=vocab_size,
                               rate=dropout_rate)
        self.fc1 = tf.keras.layers.Dense(d_model, activation=tf.keras.activations.gelu)
        self.layernorm = tf.keras.layers.LayerNormalization(axis=-1)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=None, mask=None):
        x, adjoin_matrix, attn_mask = inputs
        x = self.encoder([x, adjoin_matrix, attn_mask], training=training)
        x = self.fc1(x)
        x = self.layernorm(x)
        x = self.fc2(x)
        return x


class ModelFinetune(tf.keras.Model):
    """
    Finetuning model for downstream property prediction.
    Used for predicting target properties from encoded input representations.

    Main parameters:
        num_layers: number of encoder blocks.
        d_model: hidden dimension of token representations.
        dff: hidden dimension of the feed-forward network.
        num_heads: number of attention heads.
        vocab_size: vocabulary size of input tokens.
        dropout_rate: dropout rate.
    """
    
    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=4, vocab_size=17, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               input_vocab_size=vocab_size,
                               rate=dropout_rate)

        self.fc1 = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(0.1))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.fc2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x, adjoin_matrix, attn_mask = inputs
        h = self.encoder([x, adjoin_matrix, attn_mask], training=training)

        x = h[:, 0, :]
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return x
