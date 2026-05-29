class VAE():

    name = 'VAE'

    config = dict(hidden_layers=1,
                  hidden_size_factor=.2,
                  latent_size_factor=.05,
                  noise=None)

    def __init__(self):
        """Initialize VAE model."""
        self.model = None

    @staticmethod
    def model_fn(dataset, **kwargs):
        # Import keras locally
        from keras.layers import Input, Dense, Dropout, GaussianNoise, Lambda
        from keras.models import Model
        from keras.optimizers import Adam
        from keras import backend as K

        hidden_layers = kwargs.pop('hidden_layers')
        hidden_size_factor = kwargs.pop('hidden_size_factor')
        latent_size_factor = kwargs.pop('latent_size_factor')
        noise = kwargs.pop('noise')

        features = dataset.flat_onehot_features_2d
        input_size = features.shape[1]

        # 1. Encoder
        inputs = Input(shape=(input_size,), name='input')
        x = inputs

        if noise is not None:
            x = GaussianNoise(noise)(x)

        for i in range(hidden_layers):
            if isinstance(hidden_size_factor, list):
                factor = hidden_size_factor[i]
            else:
                factor = hidden_size_factor
            x = Dense(int(input_size * factor), activation='relu', name=f'enc_hid{i + 1}')(x)
            x = Dropout(0.5)(x)

        # 2. Latent Space (Z)
        latent_dim = max(2, int(input_size * latent_size_factor))
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var_raw = Dense(latent_dim, name='z_log_var')(x)

        # 🚨 终极防爆：强行把 log_var 截断在 -10 到 10 之间，彻底杜绝 e^x 内存溢出！
        z_log_var = Lambda(lambda v: K.clip(v, -10.0, 10.0), name='z_log_var_clipped')(z_log_var_raw)

        # Reparameterization Trick 采样函数
        def sampling(args):
            z_m, z_l_v = args
            batch = K.shape(z_m)[0]
            dim = K.int_shape(z_m)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_m + K.exp(0.5 * z_l_v) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # 3. Decoder
        x_dec = z
        for i in range(hidden_layers):
            if isinstance(hidden_size_factor, list):
                factor = hidden_size_factor[-(i + 1)]
            else:
                factor = hidden_size_factor
            x_dec = Dense(int(input_size * factor), activation='relu', name=f'dec_hid{i + 1}')(x_dec)
            x_dec = Dropout(0.5)(x_dec)

        outputs = Dense(input_size, activation='linear', name='output')(x_dec)

        # Build model
        model = Model(inputs=inputs, outputs=outputs)

        # 4. Custom Loss: Reconstruction (MSE) + KL Divergence
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        # 配合权重压制
        kl_loss = K.mean(kl_loss) / (input_size * 1000.0)
        model.add_loss(kl_loss)

        # Compile model
        model.compile(
            optimizer=Adam(lr=0.00005, beta_2=0.99, clipvalue=1.0),
            loss='mean_squared_error',
        )

        return model, features, features

    def detect(self, dataset):
        import numpy as np

        _, features, _ = self.model_fn(dataset, **self.config)

        input_size = int(self.model.input.shape[1])
        features_size = int(features.shape[1])
        if input_size > features_size:
            features = np.pad(features, [(0, 0), (0, input_size - features_size), (0, 0)], mode='constant')
        elif input_size < features_size:
            features = features[:, :input_size]

        predictions = self.model.predict(features)
        errors = np.power(features - predictions, 2)

        split = np.cumsum(np.tile(dataset.attribute_dims, [dataset.max_len]), dtype=int)[:-1]
        errors = np.split(errors, split, axis=1)
        errors = np.array([np.mean(a, axis=1) if len(a) > 0 else 0.0 for a in errors])

        scores = np.zeros((len(dataset.y), dataset.max_len, len(dataset.attribute_dims)))
        for i in range(len(dataset.attribute_dims)):
            error = errors[i::len(dataset.attribute_dims)]
            scores[:, :, i] = error.T

        num_groups = int(dataset.mask.shape[1] / sum(dataset.attribute_dims))
        grouped_mask = np.zeros((dataset.mask.shape[0], num_groups), dtype=bool)

        for i in range(num_groups):
            start = i * sum(dataset.attribute_dims)
            end = (i + 1) * sum(dataset.attribute_dims)
            grouped_mask[:, i] = dataset.mask.iloc[:, start:end].any(axis=1)

        mean_scores = np.mean(scores, axis=2)
        mean_scores = mean_scores.flatten()
        grouped_mask = grouped_mask.flatten()
        mean_scores = mean_scores[~grouped_mask]

        return mean_scores