import sys
sys.path.append('/trading_robot/')
from exports.Preprocessing import Preprocssing_initiation
import torch.nn.functional as F
from utils.functions.train import train_classification
from utils.functions.hetvae import Namespace
import warnings



warnings.filterwarnings('ignore')

# data_args = {}
# data_args['max_len'] = 40
# data_args['vars'] = {'X': ['Open', 'High', 'Low', 'Close', 'Volume']}
# data_args['normalize'] = True
# data_args['scaling_model_name'] = 'test_OHLC_Daily_Scaling_Model'
# data_args['batch_size'] = 64
# data_args['train_size'] = 0.9
# data_args['min_filled_seq_len'] = 40
# data_args['sample_number_of_tickers'] = 2000
# data_args['number_of_sample'] = 100000
# data_args['shuffle'] = True
# data_args['remove_outlier_cols'] = ['Close_r']
# data_args['compute_return_cols'] = ['Open', 'High', 'Low', 'Close', 'Volume']
# data_args['genereate_technical_indicator'] = True
# data_args['remove_outliers'] = True
# data_args['box_cox_vars'] = ['Open', 'High', 'Low', 'Close', 'Volume']
# data_args['multitask_args'] = {
#     'IN':  {'Y_clas': [
#         ['Close', 10, 5, 3],
#         ['Close', 12, 3, 3],
#         ['Close', 14, 2, 3],
#         ['Close', 16, 1, 3],
#         ['High', 16, 1, 3],
#         ['Low', 16, 1, 3],
#         ['Open', 16, 1, 3],
#         ['Volume', 16, 1, 3],
#         ['Volume', 14, 2, 3],
#         ['Volume', 12, 3, 3],]}
# }


training_args = {
    'NumberOfTickerInTrainingSample': 2000,
    'number_of_sample': 150000,
    'min_filled_seq_len': 60,
    'max_len': 60,
    'batch_size': 256,
    'train_size': 0.8,
    'shuffle': True,
}


# training_args = {
#     'NumberOfTickerInTrainingSample': 20,
#     'number_of_sample': 1000,
#     'min_filled_seq_len': 60,
#     'max_len': 60,
#     'batch_size': 256,
#     'train_size': 0.8,
#     'shuffle': True,
# }

# Preprocssing_initiation.Preporcessing_initiation(data_args, 'DailyOhlcPreprocessing')


model_args_classifier = Namespace(batch_size=training_args['batch_size'], bound_variance=True, const_var=False, model_name='Daily_Hetvae', dropout=0.0,
                                  elbo_weight=1.0, embed_time=256, enc_num_heads=1, intensity=True, k_iwae=1, kl_annealing=False,
                                  kl_zero=False, latent_dim=64, lr=0.001, mixing='concat', mse_weight=3.0, sample_nb=1, net='hetvae_classifier',
                                  niters=20, norm=True, normalize_input='znorm', num_ref_points=16, rec_hidden=256, recon_loss=False,
                                  sample_tp=1, seed=1, shuffle=True, std=0.1, var_per_dim=False, width=512, nb_iter_tosave=1,
                                  classification=True, class_weight=1.0, save=True)

out = train_classification('MinuteOhlcPreprocessing',
                           training_args, model_args_classifier, detailed_print=False)
