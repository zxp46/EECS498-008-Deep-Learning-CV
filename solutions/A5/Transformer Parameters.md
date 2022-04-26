# Transformer Parameters

num_heads = 4

emb_dim = 64

dim_feedforward = 256

dropout = 0.1 #Dropout is crucial

num_enc_layers = 2

num_dec_layers = 2

vocab_len = len(vocab)

loss_func = LabelSmoothingLoss

poss_enc = position_encoding_sinusoid

num_epochs = 500

warmup_interval = 20

lr = 8e-4
