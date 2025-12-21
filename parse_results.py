import numpy as np

# print floats without scientific notation and with reasonable precision
np.set_printoptions(suppress=True, precision=6)

preds = np.load('/projects/prjs1859/ICML25-TimeVLM/results/long_term_forecast_clip_Traffic_512_720_TimeVLM_custom_ftM_sl512_ll48_pl720_dm512_fs1.0_0/pred.npy')      # -> (N,pred_len,F)
trues = np.load('/projects/prjs1859/ICML25-TimeVLM/results/long_term_forecast_clip_Traffic_512_720_TimeVLM_custom_ftM_sl512_ll48_pl720_dm512_fs1.0_0/true.npy')     # -> (N,pred_len,F)
metrics = np.load('/projects/prjs1859/ICML25-TimeVLM/results/long_term_forecast_clip_Traffic_512_720_TimeVLM_custom_ftM_sl512_ll48_pl720_dm512_fs1.0_0/metrics.npy')# -> [mae,mse,rmse,mape,mspe]

print("preds.shape:", preds.shape)
print("trues.shape:", trues.shape)
print("metrics (mae,mse,rmse,mape,mspe):", metrics)
