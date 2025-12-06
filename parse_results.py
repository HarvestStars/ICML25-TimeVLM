import numpy as np

preds = np.load('results/long_term_forecast_clip_ETTh1_512_96_TimeVLM_ETTh1_ftM_sl512_ll48_pl96_dm32_fs1.0_0/pred.npy')      # -> (N,pred_len,F)
trues = np.load('results/long_term_forecast_clip_ETTh1_512_96_TimeVLM_ETTh1_ftM_sl512_ll48_pl96_dm32_fs1.0_0/true.npy')     # -> (N,pred_len,F)
metrics = np.load('results/long_term_forecast_clip_ETTh1_512_96_TimeVLM_ETTh1_ftM_sl512_ll48_pl96_dm32_fs1.0_0/metrics.npy')# -> [mae,mse,rmse,mape,mspe]

print("preds.shape:", preds.shape)
print("trues.shape:", trues.shape)
print("metrics (mae,mse,rmse,mape,mspe):", metrics)
