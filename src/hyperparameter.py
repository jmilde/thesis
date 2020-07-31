from os.path import expanduser
import tensorflow as tf
params = {
    "256":{
        "path_ckpt": expanduser('~/models/'),
        "path_cond": expanduser('~/data/eudata_conditionals.npz'),
        "path_data": expanduser('~/data/imgs'),
        "path_log": expanduser("~/cache/tensorboard-logdir/"),
        "path_spm": expanduser("~/data/logo_vocab"),
        "path_fid": expanduser("~/data/fid/"),

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
        "fid_samples": 50000,
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

    "train":{
        #"path_ckpt": expanduser('~/data/models/'),
        #"path_cond": expanduser('~/data/eudata_conditionals.npz'),
        #"path_data": expanduser('~/data/imgs'),
        #"path_log": expanduser("~/cache/tensorboard-logdir/"),
        #"path_spm": expanduser("~/data/logo_vocab"),
        #"path_fid": expanduser("~/data/fid/"),
        #"path_inception": expanduser("~/data/"),

        "path_ckpt": expanduser('~/'),
        "path_cond": expanduser('~/eudata_conditionals.npz'),
        "path_data": expanduser('~/imgs'),
        "path_log": expanduser("~/"),
        "path_spm": expanduser("~/logo_vocab"),
        "path_fid": expanduser("~/fid/"),
        "path_inception": expanduser("~/"),

        "restore_model":False, #empty or modelname for model stored at path_ckpt
        "color_cond_type": None, #"one_hot", # "continuous"
        "txt_cond_type": None, #"rnn" #"bert"
        "normalize": False,
        # parameters
        "img_dim": [128,128,3],
        "btlnk": 256,
        "channels": [32, 64, 128, 256, 512],
        "cond_dim_color": 128, #64 #512
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
        "fid_samples_nr": 50000,
        'normalizer_enc': tf.keras.layers.BatchNormalization,
        'normalizer_dec': tf.keras.layers.BatchNormalization,

        ### loss weights
        "weight_rec": 0.5, # beta: og:0.5, 0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
        "weight_kl": 1,
        "weight_neg": 0.25, # alpha: og:0.25, 0.1-0.5
        "m_plus": 180, #og:110, #250 should be selected according to the value of β, to balance advaserial loss
        "lr_enc": 0.0002, #0.0002,
        "lr_dec": 0.0002, #0.0002,
        "beta1": 0.9,
        "beta2": 0.999},

    "for_flask":{
        "path_ckpt": expanduser('~/models/'),
        "path_cond": expanduser('~/data/eudata_conditionals.npz'),
        "path_data": expanduser('~/data/imgs'),
        "path_log": expanduser("~/cache/tensorboard-logdir/"),
        "path_spm": expanduser("~/data/logo_vocab"),

        "restore_model":False, #empty or modelname for model stored at path_ckpt
        "color_cond_type": None, #"one_hot", # "continuous"
        "txt_cond_type": None, #"rnn"
        # parameters
        "img_dim": [128,128,3],
        "btlnk": 256,
        "channels": [32, 64, 128, 256, 512],
        "cond_dim_color": 128, #64 #512
        "rnn_dim": 128, # output= dimx2 because of bidirectional concat
        "cond_dim_txts": 256,
        "emb_dim": 128,
        "dropout_conditionals":0,
        "dropout_encoder_resblock":0,
        "noise_color": 0,
        "noise_txt": 0,
        "noise_img": 0,

        # training
        "vae_epochs": 20, # pretrain only vae
        "epochs": 0,
        "batch_size": 64,
        "logs_per_epoch": 100,  # log ~100x per epoch
        'normalizer_enc': tf.keras.layers.BatchNormalization,
        'normalizer_dec': tf.keras.layers.BatchNormalization,

        ### loss weights
        "weight_rec": 0.5, # beta: og:0.5, 0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
        "weight_kl": 1,
        "weight_neg": 0.25, # alpha: og:0.25, 0.1-0.5
        "m_plus": 110, #og:110, #250 should be selected according to the value of β, to balance advaserial loss
        "lr_enc": 0.0002, #0.0002,
        "lr_dec": 0.0002, #0.0002,
        "beta1": 0.9,
        "beta2": 0.999},
}
