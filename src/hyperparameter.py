from os.path import expanduser
import tensorflow as tf
import tensorflow_addons as tfa
params = {
    "256":{
        "path_ckpt": expanduser('~/models/'),
        "path_cond": expanduser('~/data/eudata_conditionals.npz'),
        "path_data": expanduser('~/data/imgs'),
        "path_log": expanduser("~/cache/tensorboard-logdir/"),
        "path_spm": expanduser("~/data/logo_vocab"),
        "path_fid": expanduser("~/data/fid/"),

        "restore_model":True, #empty or modelname for model stored at path_ckpt

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
        "vae_epochs": 0, # pretrain only vae
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

    "dataset": {"only":{"path_ckpt": expanduser('/home/jan/data/models/'),
                        "path_cond": expanduser('/home/jan/data/onlylogo_conditionals.npz'),
                        "path_data": expanduser('/home/jan/data/only_logo'),
                        "path_log": expanduser("/home/jan/cache/tensorboard-logdir/"),
                        "path_spm": expanduser("/home/jan/data/logo_vocab"),
                        "path_fid": expanduser("/home/jan/data/fid/"),
                        "path_fid_dataset": expanduser("/home/jan/data/fid/mu_var_datasetonly.npz"),
                        "path_inception": expanduser("/home/jan/data/"),},
                "lld":{"path_ckpt": expanduser('/home/jan/data/models/'),
                       "path_cond": expanduser('/home/jan/data/lldboosted_conditionals.npz'),
                       "path_data": expanduser('/home/jan/data/lld_boosted'),
                       "path_log": expanduser("/home/jan/cache/tensorboard-logdir/"),
                       "path_spm": expanduser("/home/jan/data/logo_vocab"),
                       "path_fid": expanduser("/home/jan/data/fid/"),
                       "path_fid_dataset": expanduser("/home/jan/data/fid/mu_var_datasetlld.npz"),
                       "path_inception": expanduser("/home/jan/data/"),},
                "all":{"path_ckpt": expanduser('/home/jan/data/models/'),
                       "path_cond": expanduser('/home/jan/data/eudata_conditionals.npz'),
                       "path_data": expanduser('/home/jan/data/imgs'),
                       "path_log": expanduser("/home/jan/cache/tensorboard-logdir/"),
                       "path_spm": expanduser("/home/jan/data/logo_vocab"),
                       "path_fid": expanduser("/home/jan/data/fid/"),
                       "path_fid_dataset": expanduser("/home/jan/data/fid/mu_var_dataset.npz"),
                       "path_inception": expanduser("/home/jan/data/"),}},

    "train_0":{
        "gpu":0,
        "model_name": None,
        "dataset": "all", # "all" "only" "lld"
        "restore_model": True, #empty or modelname for model stored at path_ckpt
        "cond_model": "dec", # None, "dec", "enc_dec"
        "auxilary": False,
        "color_cond_type": None, #"one_hot", # "continuous"
        "txt_cond_type": "rnn", #"rnn" #"bert"
        "cluster_cond_type": None, #"vgg"
        "txt_len_min":1,
        "txt_len_max":5,
        "normalize": True,

        # parameters
        "img_dim": [128,128,3],
        "btlnk": 256,
        "channels": [32, 64, 128, 256, 512],
        "cond_dim_color": 128, #64 #512
        "rnn_dim": 128, # output= dimx2 because of bidirectional concat
        "cond_dim_txts": 256,
        "cond_dim_clusters": 128,
        "emb_dim": 128,
        "dropout_conditionals":0,
        "dropout_encoder_resblock":0,
        "noise_color": 0,
        "noise_txt": 0,
        "noise_img": 0,

        # training
        "vae_epochs": 0, # pretrain only vae
        "epochs": 10, #lld:276, all:20, only:115 ,txt:50
        "batch_size": 60,
        "logs_per_epoch": 100,  # log ~100x per epoch
        "fid_samples_nr": 25000,
        "plot_bn"       : False,
        'normalizer_enc': "batch", #None "instance" "batch" "group" "layer"
        'normalizer_dec': "batch", #None "instance" "batch" "group" "layer"

        ### loss weights
        "weight_rec": 0.25, # beta: og:0.5, 0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
        "weight_kl": 1,
        "weight_neg": 0.25, # alpha: og:0.25, 0.1-0.5
        "weight_aux": 25,
        "m_plus": 110, #og:110, #250 should be selected according to the value of β, to balance advaserial loss
        "lr_enc": 0.0001, #0.0002,
        "lr_dec": 0.0001, #0.0002,
        "beta1": 0.9,
        "beta2": 0.999},

        "train_1":{
            "gpu":1,
            "model_name": None,
            "dataset": "lld", # "all" "only" "lld"
            "restore_model": True, #empty or modelname for model stored at path_ckpt
            "cond_model": "dec", # None, "dec", "enc_dec"
            "auxilary": True,
            "color_cond_type": "one_hot", #"one_hot", # "continuous"
            "txt_cond_type": None, #"rnn" #"bert"
            "cluster_cond_type": None, #"vgg"
            "txt_len_min":0,
            "txt_len_max":1000,
            "normalize": True,

            # parameters
            "img_dim": [128,128,3],
            "btlnk": 256,
            "channels": [32, 64, 128, 256, 512],
            "cond_dim_color": 128, #64 #512
            "rnn_dim": 128, # output= dimx2 because of bidirectional concat
            "cond_dim_txts": 256,
            "cond_dim_clusters": 128,
            "emb_dim": 128,
            "dropout_conditionals":0,
            "dropout_encoder_resblock":0,
            "noise_color": 0,
            "noise_txt": 0,
            "noise_img": 0,

            # training
            "vae_epochs": 0, # pretrain only vae
            "epochs": 10, #lld:276, all:20, only:115 ,txt:50
            "batch_size": 60,
            "logs_per_epoch": 100,  # log ~100x per epoch
            "fid_samples_nr": 25000,
            "plot_bn"       : False,
            'normalizer_enc': "batch", #None "instance" "batch" "group" "layer"
            'normalizer_dec': "batch", #None "instance" "batch" "group" "layer"

            ### loss weights
            "weight_rec": 0.25, # beta: og:0.5, 0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
            "weight_kl": 1,
            "weight_neg": 0.25, # alpha: og:0.25, 0.1-0.5
            "weight_aux": 50,
            "m_plus": 110, #og:110, #250 should be selected according to the value of β, to balance advaserial loss
            "lr_enc": 0.0001, #0.0002,
            "lr_dec": 0.0001, #0.0002,
            "beta1": 0.9,
            "beta2": 0.999},

    "train_2":{
            "gpu":2,
            "model_name": None,
            "dataset": "lld", # "all" "only" "lld"
            "restore_model": True, #empty or modelname for model stored at path_ckpt
            "cond_model": "dec", # None, "dec", "enc_dec"
            "auxilary": True,
            "color_cond_type": "one_hot", #"one_hot", # "continuous"
            "txt_cond_type": None, #"rnn" #"bert"
            "cluster_cond_type": None, #"vgg"
            "txt_len_min":0,
            "txt_len_max":1000,
            "normalize": True,

            # parameters
            "img_dim": [128,128,3],
            "btlnk": 256,
            "channels": [32, 64, 128, 256, 512],
            "cond_dim_color": 128, #64 #512
            "rnn_dim": 128, # output= dimx2 because of bidirectional concat
            "cond_dim_txts": 256,
            "cond_dim_clusters": 128,
            "emb_dim": 128,
            "dropout_conditionals":0,
            "dropout_encoder_resblock":0,
            "noise_color": 0,
            "noise_txt": 0,
            "noise_img": 0,

            # training
            "vae_epochs": 0, # pretrain only vae
            "epochs": 10, #lld:276, all:20, only:115 ,txt:50
            "batch_size": 60,
            "logs_per_epoch": 100,  # log ~100x per epoch
            "fid_samples_nr": 25000,
            "plot_bn"       : False,
            'normalizer_enc': "batch", #None "instance" "batch" "group" "layer"
            'normalizer_dec': "batch", #None "instance" "batch" "group" "layer"

            ### loss weights
            "weight_rec": 0.25, # beta: og:0.5, 0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
            "weight_kl": 1,
            "weight_neg": 0.25, # alpha: og:0.25, 0.1-0.5
            "weight_aux": 25,
            "m_plus": 110, #og:110, #250 should be selected according to the value of β, to balance advaserial loss
            "lr_enc": 0.0001, #0.0002,
            "lr_dec": 0.0001, #0.0002,
            "beta1": 0.9,
            "beta2": 0.999},

    "train_3":{
            "gpu":3,
            "model_name": "INTRO_all_BATCH__20-m110-lr0.0001b10.9b20.999-w_rec0.25-w_neg0.25-lr0.0001-z256dec-txt:(bert-dense256-rnn128-emb128-1<10)-128,128,3",
            "dataset": "all", # "all" "only" "lld"
            "restore_model": True, #empty or modelname for model stored at path_ckpt
            "cond_model": "dec", # None, "dec", "enc_dec"
            "auxilary": False,
            "color_cond_type": None , #"one_hot", # "continuous"
            "txt_cond_type": "bert", #"rnn" #"bert"
            "cluster_cond_type": None, #"vgg"
            "txt_len_min":1,
            "txt_len_max":10,
            "normalize": True,

            # parameters
            "img_dim": [128,128,3],
            "btlnk": 256,
            "channels": [32, 64, 128, 256, 512],
            "cond_dim_color": 128, #64 #512
            "rnn_dim": 128, # output= dimx2 because of bidirectional concat
            "cond_dim_txts": 256,
            "cond_dim_clusters": 128,
            "emb_dim": 128,
            "dropout_conditionals":0,
            "dropout_encoder_resblock":0,
            "noise_color": 0,
            "noise_txt": 0,
            "noise_img": 0,

            # training
            "vae_epochs": 0, # pretrain only vae
            "epochs": 0, #lld:276, all:20, only:115 ,txt:50
            "batch_size": 60,
            "logs_per_epoch": 100,  # log ~100x per epoch
            "fid_samples_nr": 25000,
            "plot_bn"       : False,
            'normalizer_enc': "batch", #None "instance" "batch" "group" "layer"
            'normalizer_dec': "batch", #None "instance" "batch" "group" "layer"

            ### loss weights
            "weight_rec": 0.25, # beta: og:0.5, 0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
            "weight_kl": 1,
            "weight_neg": 0.25, # alpha: og:0.25, 0.1-0.5
            "weight_aux": 10,
            "m_plus": 110, #og:110, #250 should be selected according to the value of β, to balance advaserial loss
            "lr_enc": 0.0001, #0.0002,
            "lr_dec": 0.0001, #0.0002,
            "beta1": 0.9,
            "beta2": 0.999},


    "for_flask":{
        "gpu": 3,
        "model_name": "INTRO_all_BATCH__50-m110-lr0.0002b10.9b20.999-w_rec0.25-w_neg0.25-lr0.0002-z256dec-txt:(rnn-dense256-rnn128-emb128-1<10)-128,128,3",
        "start_step": 0,
        "dataset": "lld", # "all" "only" "lld"
        "restore_model": True, #empty or modelname for model stored at path_ckpt
        "cond_model": "dec", # None, "dec", "enc_dec"
        "auxilary": False,
        "color_cond_type": None, #"one_hot", # "continuous"
        "txt_cond_type": "rnn", #"rnn" #"bert"
        "cluster_cond_type": None, #"vgg"
        "txt_len_min":0,
        "txt_len_max":1000,
        "normalize": True,


        # parameters
        "img_dim": [128,128,3],
        "btlnk": 256,
        "channels": [32, 64, 128, 256, 512],
        "cond_dim_color": 128, #64 #512
        "rnn_dim": 128, # output= dimx2 because of bidirectional concat
        "cond_dim_txts": 256,
        "cond_dim_clusters": 128,
        "emb_dim": 128,
        "dropout_conditionals":0,
        "dropout_encoder_resblock":0,
        "noise_color": 0,
        "noise_txt": 0,
        "noise_img": 0,

        # training
        "vae_epochs": 0, # pretrain only vae
        "epochs": 0, #lld:276, all:20, only:115 ,txt:50
        "batch_size": 60,
        "logs_per_epoch": 100,  # log ~100x per epoch
        "fid_samples_nr": 25000,
        "plot_bn"       : False,
        'normalizer_enc': "batch", #None "instance" "batch" "group" "layer"
        'normalizer_dec': "batch", #None "instance" "batch" "group" "layer"

        ### loss weights
        "weight_rec": 0.25, # beta: og:0.5, 0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
        "weight_kl": 1,
        "weight_neg": 0.25, # alpha: og:0.25, 0.1-0.5
        "weight_aux": 100,
        "m_plus": 110, #og:110, #250 should be selected according to the value of β, to balance advaserial loss
        "lr_enc": 0.0001, #0.0002,
        "lr_dec": 0.0001, #0.0002,
        "beta1": 0.9,
        "beta2": 0.999}
}
