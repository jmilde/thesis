from os.path import expanduser
import tensorflow as tf
import tensorflow_addons as tfa
params = {

    "dataset": {"lld":{"path_ckpt": expanduser('./data/models/'),
                       "path_cond": expanduser('./data/lldboosted_conditionals.npz'),
                       "path_data": expanduser('./data/lld_boosted'),
                       "path_log": expanduser("./cache/tensorboard-logdir/"),
                       "path_spm": expanduser("./data/logo_vocab"),
                       "path_fid": expanduser("./data/fid/"),
                       "path_fid_dataset": expanduser("./data/fid/mu_var_datasetlld.npz"),
                       "path_inception": expanduser("./data/"),},
                "all":{"path_ckpt": expanduser('./data/models/'),
                       "path_cond": expanduser('./data/eudata_conditionals.npz'),
                       "path_data": expanduser('./data/imgs'),
                       "path_log": expanduser("./cache/tensorboard-logdir/"),
                       "path_spm": expanduser("./data/logo_vocab"),
                       "path_fid": expanduser("./data/fid/"),
                       "path_fid_dataset": expanduser("./data/fid/mu_var_dataset.npz"),
                       "path_inception": expanduser("./data/"),}},

    "train_0":{
        "gpu":0,
        "model_name": None, # if no name is given, then an automatic nameis picked
        "dataset": "lld", # choose between: "all" and "lld"
        "restore_model": False, # if True, then it trys to load a model with model_name
        "cond_model": "dec", # chooses from:  None, "dec", "enc_dec"
        "auxilary": False, # if true then cond_model has to be "dec" and the conditionals  need to be "one_hot" for color or "vgg" for cluster
        "color_cond_type": None, # choose from: None, "one_hot", "continuous"
        "txt_cond_type": None, # choose from: "rnn", "bert"
        "cluster_cond_type":None, # chooses from : None, "vgg"
        "txt_len_min":0, #restricts the text length, if >0 logos without text are excluded
        "txt_len_max":1000,
        "normalize": True,
        "truncate":None, # experimental, for using a truncated gaussian normal with +/- value given here as cutoff.

        # parameters
        "img_dim": [128,128,3], # dimension of the image
        "btlnk": 256,  # the bottleneck/laten spacesize
        "channels": [32, 64, 128, 256, 512], # the number of channels per residual block, is mirrored for the decoder
        "cond_dim_color": 128, # size of dense layer for the color conditipnals
        "rnn_dim": 128, # output of the RNN= dim*2 because of bidirectional concat
        "emb_dim": 128, # size of the embeddings for the SentencePiece segments
        "cond_dim_txts": 256, # size of dense layer for the text conditoinals
        "cond_dim_clusters": 128, # size of dense layer for the cluster conditionals
        "dropout_conditionals":0,
        "dropout_encoder_resblock":0,
        "noise_color": 0,
        "noise_txt": 0,
        "noise_img": 0,

        # training
        "vae_epochs": 0, # train the model as a VAE, can be used for pretraining
        "epochs": 0, #lld:276, all:20
        "batch_size": 60,
        "logs_per_epoch": 100,  # log ~100x per epoch
        "fid_samples_nr": 25000,
        "plot_bn"       : False, # plot with batch normalization used as if training
        'normalizer_enc': "batch", # choose from: None "instance" "batch" "group" "layer"
        'normalizer_dec': "batch", # choose from: None "instance" "batch" "group" "layer"

        ### loss weights
        "weight_rec": 0.1, # beta: 0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
        "weight_kl": 1, # keep this fixed
        "weight_neg": 0.25, # alpha: 0.1-0.5, best to keep fixed and to only vary weight_rec and m_plus
        "weight_aux": 10, # weight for the auxilary loss 10-25 recommended
        "m_plus": 175, #~110-250 should be selected according to the value of β, to balance advesarial loss
        "lr_enc": 0.0001, # 0.0002,
        "lr_dec": 0.0001, # 0.0002,
        "beta1": 0.9,
        "beta2": 0.999},

    "for_flask":{
        "gpu":0,
        "model_name": "yyINTRO_lld_BATCH__10-m175-lr0.0001b10.9b20.999-w_rec0.1-w_neg0.25-lr0.0001-z256dec-color:(continuous128)-128,128,3",
        "dataset": "lld", # "all" "only" "lld"
        "restore_model": True, #empty or modelname for model stored at path_ckpt
        "cond_model": "dec", # None, "dec", "enc_dec"
        "auxilary": False,
        "color_cond_type": "continuous" , #"one_hot", # "continuous"
        "txt_cond_type": None, #"rnn" #"bert"
        "cluster_cond_type": None, #"vgg"
        "txt_len_min":0,
        "txt_len_max":1000,
        "normalize": True,
        "truncate":None,

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
        "weight_rec": 0.1, # beta: og:0.5, 0.01 - 100, larger β improves reconstruction quality but may influence sample diversity
        "weight_kl": 1,
        "weight_neg": 0.25, # alpha: og:0.25, 0.1-0.5
        "weight_aux": 25,
        "m_plus": 175, #og:110, #250 should be selected according to the value of β, to balance advaserial loss
        "lr_enc": 0.0001, #0.0002,
        "lr_dec": 0.0001, #0.0002,
        "beta1": 0.9,
        "beta2": 0.999}
}
