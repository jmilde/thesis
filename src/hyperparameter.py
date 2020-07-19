from os.path import expanduser
import tensorflow as tf
params = {
    "256":{
        "path_ckpt": expanduser('~/models/'),
        "path_cond": expanduser('~/data/eudata_conditionals.npz'),
        "path_data": expanduser('~/data/imgs'),
        "path_log": expanduser("~/cache/tensorboard-logdir/"),
        "path_spm": expanduser("~/data/logo_vocab"),

        "restore_model":False, #empty or modelname for model stored at path_ckpt

        # Data info
        "img_dim": [256,256,3],
        "btlnk": 512,
        "channels": [32, 64, 128, 256, 512, 512],
        "cond_dim_color": 256, #64 #512
        "rnn_dim": 128, # output= dimx2 because of bidirectional concat
        "cond_dim_txts": 256,
        "emb_dim": 128,
        "dropout_conditionals":0.3,
        "dropout_encoder_resblock":0.3,
        "noise_color": 0.1,
        "noise_txt": 0.1,
        "noise_img": 0.1,

        # training
        "vae_epochs": 50, # pretrain only vae
        "epochs": 0,
        "batch_size": 16,
        "logs_per_epoch": 100,  # log ~100x per epoch
        'normalizer_enc': tf.keras.layers.BatchNormalization,
        'normalizer_dec': tf.keras.layers.BatchNormalization,


        ### loss weights
        "weight_rec": 0.2, #beta  0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
        "weight_kl": 1,
        "weight_neg": 0.5, #alpha 0.1-0.5
        "m_plus": 500, # should be selected according to the value of β, to balance advaserial loss
        "lr_enc":0.00005,
        "lr_dec":0.00005,
        "beta1": 0.9},
    "Intro":{
        "path_ckpt": expanduser('~/models/'),
        "path_cond": expanduser('~/data/eudata_conditionals.npz'),
        "path_data": expanduser('~/data/imgs'),
        "path_log": expanduser("~/cache/tensorboard-logdir/"),
        "path_spm": expanduser("~/data/logo_vocab"),

        "restore_model":False, #empty or modelname for model stored at path_ckpt

        # parameters
        "img_dim": [128,128,3],
        "btlnk": 256,
        "channels": [32, 64, 128, 256, 512],
        "cond_dim_color": 256, #64 #512
        "rnn_dim": 128, # output= dimx2 because of bidirectional concat
        "cond_dim_txts": 256,
        "emb_dim": 128,
        "dropout_conditionals":0,
        "dropout_encoder_resblock":0,
        "noise_color": 0,
        "noise_txt": 0,
        "noise_img": 0,

        # training
        "vae_epochs": 0, # pretrain only vae
        "epochs": 20,
        "batch_size": 64,
        "logs_per_epoch": 100,  # log ~100x per epoch
        'normalizer_enc': tf.keras.layers.BatchNormalization,
        'normalizer_dec': tf.keras.layers.BatchNormalization,

        ### loss weights
        "weight_rec": 0.2, #beta  0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
        "weight_kl": 1,
        "weight_neg": 0.5, #alpha 0.1-0.5
        "m_plus": 250, # should be selected according to the value of β, to balance advaserial loss
        "lr_enc":0.0002,
        "lr_dec":0.0002,
        "beta1": 0.9},

    "adding":{
        "path_ckpt": expanduser('~/models/'),
        "path_cond": expanduser('~/data/eudata_conditionals.npz'),
        "path_data": expanduser('~/data/imgs'),
        "path_log": expanduser("~/cache/tensorboard-logdir/"),
        "path_spm": expanduser("~/data/logo_vocab"),

        "restore_model":False, #empty or modelname for model stored at path_ckpt

        # parameters
        "img_dim": [128,128,3],
        "btlnk": 256,
        "channels": [32, 64, 128, 256, 512],
        "cond_dim_color": 256, #64 #512
        "rnn_dim": 128, # output= dimx2 because of bidirectional concat
        "cond_dim_txts": 256,
        "emb_dim": 128,
        "dropout_conditionals":0,
        "dropout_encoder_resblock":0,
        "noise_color": 0,
        "noise_txt": 0,
        "noise_img": 0.1,

        # training
        "vae_epochs": 20, # pretrain only vae
        "epochs": 0,
        "batch_size": 96,
        "logs_per_epoch": 100,  # log ~100x per epoch
        'normalizer_enc': tf.keras.layers.BatchNormalization,
        'normalizer_dec': tf.keras.layers.BatchNormalization,

        ### loss weights
        "weight_rec": 0.2, #beta  0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
        "weight_kl": 1,
        "weight_neg": 0.5, #alpha 0.1-0.5
        "m_plus": 500, # should be selected according to the value of β, to balance advaserial loss
        "lr_enc":0.0001,
        "lr_dec":0.0001,
        "beta1": 0.9},

    "V2AE_RNNcolor128txts128-pre50-128,128,3-m500-lr5e-05b0.9-w_rec0.2-rnn128-emb128":{
        "path_ckpt": expanduser('~/models/'),
        "path_cond": expanduser('~/data/eudata_conditionals.npz'),
        "path_data": expanduser('~/data/imgs'),
        "path_log": expanduser("~/cache/tensorboard-logdir/"),
        "path_spm": expanduser("~/data/logo_vocab"),

        "restore_model":False, #empty or modelname for model stored at path_ckpt

        # parameters
        "img_dim": [128,128,3],
        "btlnk": 256,
        "channels": [32, 64, 128, 256, 512],
        "cond_dim_color": 256, #64 #512
        "rnn_dim": 128, # output= dimx2 because of bidirectional concat
        "cond_dim_txts": 256,
        "emb_dim": 128,
        "dropout_conditionals":0.3,
        "dropout_encoder_resblock":0.3,
        "noise_color": 0.1,
        "noise_txt": 0.1,
        "noise_img": 0.1,

        # training
        "vae_epochs": 50, # pretrain only vae
        "epochs": 0,
        "batch_size": 96,
        "logs_per_epoch": 100,  # log ~100x per epoch
        'normalizer_enc': tf.keras.layers.BatchNormalization,
        'normalizer_dec': tf.keras.layers.BatchNormalization,

        ### loss weights
        "weight_rec": 0.2, #beta  0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
        "weight_kl": 1,
        "weight_neg": 0.5, #alpha 0.1-0.5
        "m_plus": 500, # should be selected according to the value of β, to balance advaserial loss
        "lr_enc":0.00005,
        "lr_dec":0.00005,
        "beta1": 0.9},

     "for_flask" :{
         "path_ckpt": expanduser('~/models/'),
         "path_cond": expanduser('~/data/eudata_conditionals.npz'),
         "path_data": expanduser('~/data/imgs'),
         "path_log": expanduser("~/cache/tensorboard-logdir/"),
         "path_spm": expanduser("~/data/logo_vocab"),

         "restore_model":False, #empty or modelname for model stored at path_ckpt

         # parameters
         "img_dim": [128,128,3],
         "btlnk": 256,
         "channels": [32, 64, 128, 256, 512],
         "cond_dim_color": 128, #64 #512
         "rnn_dim": 128, # output= dimx2 because of bidirectional concat
         "cond_dim_txts": 128,
         "emb_dim": 128,
         "dropout_conditionals":0.5,
         "dropout_encoder_resblock":0.3,
         "noise_color": 0.1,
         "noise_txt": 0.1,
         "noise_img": 0.1,

         # training
         "vae_epochs": 50, # pretrain only vae
         "epochs": 0,
         "batch_size": 96,
         "logs_per_epoch": 100,  # log ~100x per epoch
         'normalizer_enc': tf.keras.layers.BatchNormalization,
         'normalizer_dec': tf.keras.layers.BatchNormalization,

         ### loss weights
         "weight_rec": 0.2, #beta  0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
         "weight_kl": 1,
         "weight_neg": 0.5, #alpha 0.1-0.5
         "m_plus": 500, # should be selected according to the value of β, to balance advaserial loss
         "lr_enc":0.00005,
         "lr_dec":0.00005,
         "beta1": 0.9}
}
