import sys
sys.path.append('/trading_robot/')

import warnings
from utils.functions.hetvae import Namespace
from utils.functions.train import train_classification
import torch.nn.functional as F
from exports.Preprocessing import Preprocssing_initiation

warnings.filterwarnings('ignore')



training_args = {
    'NumberOfTickerInTrainingSample' : 3000,
    'number_of_sample'               : 300000,
    'min_filled_seq_len'             : 50 ,
    'max_len'                        : 50 , 
    'batch_size'                     : 256 ,
    'train_size'                     : 0.8,
    'shuffle'                        : True,
}

model_args_classifier = Namespace(batch_size=training_args['batch_size'], bound_variance=True, const_var=False, model_name='Daily_Hetvae', dropout=0.0,
                                  elbo_weight=1.0, embed_time=256, enc_num_heads=1, intensity=True, k_iwae=1, kl_annealing=False,
                                  kl_zero=False, latent_dim=64, lr=0.001, mixing='concat', mse_weight=3.0, sample_nb=1, net='hetvae_classifier',
                                  niters=10, norm=True, normalize_input='znorm', num_ref_points=16, rec_hidden=256, recon_loss=False,
                                  sample_tp=1, seed=1, shuffle=True, std=0.1, var_per_dim=False, width=512, nb_iter_tosave=1,
                                  classification=True, class_weight=1.0, save=True)

out = train_classification('DailyOhlcPreprocessing', training_args, model_args_classifier, detailed_print=False)
