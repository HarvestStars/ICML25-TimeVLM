import os
import time
import json
import numpy as np
import torch
import pandas as pd

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.tools import visual
from utils.metrics import metric

# df_utils from your Chronos2 demo
try:
    from chronos.df_utils import convert_df_input_to_list_of_dicts_input  # type: ignore
    from chronos import Chronos2Pipeline  # type: ignore
except Exception:
    convert_df_input_to_list_of_dicts_input = None  # type: ignore
    Chronos2Pipeline = None  # type: ignore

class Exp_Chronos2_Forecast(Exp_Basic):
    """
    Pseudo-integration of Chronos2 into TimeVLM experiment framework (df_utils version).

    - Reuse VLM's data_provider / Dataset_xxx for:
        * CSV reading
        * train/val/test split
        * scaling (fit on train split)
    - Train Chronos2 using pipeline.fit() (HF Trainer inside Chronos2).
    - Evaluate using last-window, median (q=0.5) as point forecast, compute MSE/MAE.
    """

    def __init__(self, args):
        super().__init__(args)
        if Chronos2Pipeline is None:
            raise ImportError("Chronos2Pipeline import failed. Check your PYTHONPATH / install.")
        if convert_df_input_to_list_of_dicts_input is None:
            raise ImportError("convert_df_input_to_list_of_dicts_input not found. Please ensure df_utils.py is importable.")

        self.pipeline = self._build_model()

    # -------------------------- VLM required hooks --------------------------

    def _build_model(self):
        model_name_or_path = getattr(self.args, "chronos2_model", None)
        if model_name_or_path is None:
            raise ValueError("args.chronos2_model is required for Chronos2 exp.")

        device = "cuda" if getattr(self.args, "use_gpu", False) else "cpu"
        torch_dtype = getattr(self.args, "chronos2_dtype", None)

        dtype_map = {
            "float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32, "fp32": torch.float32,
        }
        dtype = dtype_map.get(str(torch_dtype).lower(), None) if torch_dtype else None

        if dtype is None:
            pipeline = Chronos2Pipeline.from_pretrained(model_name_or_path)  # type: ignore
        else:
            pipeline = Chronos2Pipeline.from_pretrained(model_name_or_path, torch_dtype=dtype)  # type: ignore

        # best-effort device placement
        if hasattr(pipeline, "to"):
            pipeline = pipeline.to(device)  # type: ignore
        return pipeline

    def _get_data(self, flag):
        data_set, _ = data_provider(self.args, flag)
        return data_set, None

    # -------------------------- Helpers --------------------------

    def _infer_pd_freq(self):
        """
        VLM args.freq often is: 'h', 't' (minute), 'd'
        pandas prefers: 'H', 'T', 'D'
        """
        freq = getattr(self.args, "freq", "h")
        freq_map = {"h": "H", "t": "T", "min": "T", "m": "T", "d": "D"}
        return freq_map.get(str(freq).lower(), freq)

    # def _get_split_timestamps(self, ds):
    #     """
    #     Prefer real timestamps if dataset exposes them; otherwise fabricate a date_range.
    #     Recommended minimal change in VLM Dataset.__read_data__:
    #         self.raw_dates = pd.to_datetime(df_raw['date'][border1:border2]).reset_index(drop=True)
    #     """
    #     if hasattr(ds, "raw_dates") and ds.raw_dates is not None:
    #         # ensure pandas datetime series
    #         print("Using real timestamps from dataset.raw_dates")
    #         return pd.to_datetime(ds.raw_dates).reset_index(drop=True)

    #     # fallback: fabricate timestamps (good enough to pass df_utils validation)
    #     T = len(ds.data_x)
    #     pd_freq = self._infer_pd_freq()
    #     return pd.date_range("2000-01-01", periods=T, freq=pd_freq)
    
    def _get_split_timestamps(self, ds):
        """
        Prefer real timestamps if dataset exposes them; otherwise fabricate a date_range.

        VLM-side robustness policy (your requested version):
        - Do NOT sort / do NOT de-duplicate / do NOT try to "fix" raw_dates.
        - If raw_dates looks problematic for pd.infer_freq (duplicate / non-monotonic / infer fails),
        then fallback to a fully fabricated regular date_range of length T.
        - Always return length == T.
        """
        T = len(ds.data_x)
        pd_freq = self._infer_pd_freq()

        def _fabricate():
            return pd.date_range("2000-01-01", periods=T, freq=pd_freq)

        # No raw_dates -> fabricate
        if not (hasattr(ds, "raw_dates") and ds.raw_dates is not None):
            return _fabricate()

        print("Using real timestamps from dataset.raw_dates")
        ts = pd.to_datetime(ds.raw_dates)

        # Basic sanity: must align length with data_x
        if len(ts) != T:
            print(f"[WARN] raw_dates length mismatch: len(raw_dates)={len(ts)} vs T={T}. Fallback to date_range.")
            return _fabricate()

        # Keep original order and length; just detect "danger" signals.
        # 1) duplicates in raw order -> infer_freq likely fails
        try:
            has_dup = pd.Series(ts).duplicated().any()
        except Exception:
            has_dup = True

        if has_dup:
            print("[WARN] raw_dates has duplicates in original order. Fallback to date_range.")
            return _fabricate()

        # 2) non-monotonic in raw order -> infer_freq likely fails (we refuse to sort)
        try:
            idx = pd.DatetimeIndex(ts)
            if not idx.is_monotonic_increasing:
                print("[WARN] raw_dates is not monotonic increasing (we do not sort). Fallback to date_range.")
                return _fabricate()
        except Exception as e:
            print(f"[WARN] DatetimeIndex conversion failed ({type(e).__name__}: {e}). Fallback to date_range.")
            return _fabricate()

        # 3) final check: infer_freq must succeed; otherwise fallback
        inferred = None
        try:
            inferred = pd.infer_freq(pd.DatetimeIndex(ts))
        except Exception as e:
            print(f"[WARN] pd.infer_freq failed ({type(e).__name__}: {e}). Fallback to date_range.")
            return _fabricate()

        if inferred is None:
            print(f"[WARN] Could not infer frequency from raw_dates. Fallback to date_range(freq={pd_freq}).")
            return _fabricate()

        # Good: return as Series aligned with T, keep original order
        return pd.Series(ts).reset_index(drop=True)

    def _build_context_df_from_split(self, ds, id_column="item_id", timestamp_column="date_id"):
        """
        Convert VLM split dataset (ds.data_x: (T,C) or (T,)) to Chronos2-style context_df:
            columns: [timestamp_column, id_column, y0, y1, ..., y{C-1}]
        where each row is a time step.
        """
        x = ds.data_x
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)

        # Ensure 2D: (T,C)
        if x.ndim == 1:
            x = x[:, None]

        # Confirm semantics: row=time, col=feature
        # This is consistent with VLM slicing seq_x = data_x[s:e] -> (seq_len,C)
        T, C = x.shape

        ts = self._get_split_timestamps(ds)
        if len(ts) != T:
            raise ValueError(f"Timestamp length mismatch: len(ts)={len(ts)} vs T={T}")

        df = pd.DataFrame({timestamp_column: ts, id_column: 1})
        target_cols = [f"y{i}" for i in range(C)]
        for i, col in enumerate(target_cols):
            df[col] = x[:, i].astype("float32")

        return df, target_cols

    def _make_last_window_from_split(self, ds, seq_len, pred_len):
        """
        Build two dataframes:
        - context_df: (seq_len rows)
        - future_df:  (pred_len rows)
        Both include timestamp + id + target columns. This matches df_utils signature.
        """
        x = ds.data_x
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        T, C = x.shape
        if T < seq_len + pred_len:
            raise ValueError(f"Not enough length: T={T}, seq_len={seq_len}, pred_len={pred_len}")

        ts = self._get_split_timestamps(ds)
        context_end = T - pred_len
        ctx_x = x[context_end - seq_len : context_end]
        fut_x = x[context_end : context_end + pred_len]
        ctx_ts = ts[context_end - seq_len : context_end]
        fut_ts = ts[context_end : context_end + pred_len]

        target_cols = [f"y{i}" for i in range(C)]

        context_df = pd.DataFrame({"date_id": ctx_ts, "item_id": 1})
        future_df  = pd.DataFrame({"date_id": fut_ts, "item_id": 1})
        for i, col in enumerate(target_cols):
            context_df[col] = ctx_x[:, i].astype("float32")
            future_df[col]  = fut_x[:, i].astype("float32")

        return context_df, future_df, target_cols

    @staticmethod
    def _mse_mae(pred: np.ndarray, true: np.ndarray):
        pred = np.asarray(pred, dtype=np.float64)
        true = np.asarray(true, dtype=np.float64)
        mse = np.mean((pred - true) ** 2)
        mae = np.mean(np.abs(pred - true))
        return mse, mae

    # -------------------------- Core: train / test --------------------------

    def train(self, setting):
        """
        Train Chronos2 via pipeline.fit(), using df_utils conversion.

        Key point:
        - We feed the whole split time series (already scaled by VLM dataset) as context_df.
        - Chronos2 internally samples windows -> no need to externally build (x,y) pairs.
        """
        train_data, _ = self._get_data(flag="train")
        val_data, _ = self._get_data(flag="val")

        pred_len = int(getattr(self.args, "pred_len"))
        context_len = int(getattr(self.args, "seq_len"))

        # Chronos2 fit knobs
        num_steps = int(getattr(self.args, "chronos2_num_steps", 1000))
        batch_size = int(getattr(self.args, "chronos2_batch_size", 256))
        lr = float(getattr(self.args, "chronos2_learning_rate", 1e-6))
        finetune_mode = getattr(self.args, "chronos2_finetune_mode", "full")
        output_dir = getattr(self.args, "checkpoints", "./checkpoints")
        logging_steps = int(getattr(self.args, "chronos2_logging_steps", 50))

        exp_dir = os.path.join(output_dir, setting)
        os.makedirs(exp_dir, exist_ok=True)

        # Save args snapshot
        try:
            with open(os.path.join(exp_dir, "chronos2_exp_args.json"), "w", encoding="utf-8") as f:
                json.dump(vars(self.args), f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # Build df inputs (whole split)
        train_df, target_cols = self._build_context_df_from_split(train_data, id_column="item_id", timestamp_column="date_id")
        val_df, _ = self._build_context_df_from_split(val_data, id_column="item_id", timestamp_column="date_id")

        # print head 10 for sanity check
        print("Train df head 10:\n", train_df.head(10))
        print("Val df head 10:\n", val_df.head(10))
        
        # Convert to list[dict] inputs as Chronos2 demo
        ft_inputs, _, _ = convert_df_input_to_list_of_dicts_input(
            df=train_df,
            future_df=None,
            id_column="item_id",
            timestamp_column="date_id",
            target_columns=target_cols,
            prediction_length=pred_len,
        )
        val_inputs, _, _ = convert_df_input_to_list_of_dicts_input(
            df=val_df,
            future_df=None,
            id_column="item_id",
            timestamp_column="date_id",
            target_columns=target_cols,
            prediction_length=pred_len,
        )

        print(f"[Chronos2] fit() start: setting={setting}")
        t0 = time.time()

        self.pipeline = self.pipeline.fit(
            inputs=ft_inputs,
            validation_inputs=val_inputs,
            prediction_length=pred_len,
            context_length=context_len,
            learning_rate=lr,
            num_steps=num_steps,
            batch_size=batch_size,
            output_dir=exp_dir,
            finetune_mode=finetune_mode,
            logging_steps=logging_steps,
        )

        dt = time.time() - t0
        print(f"[Chronos2] fit() done in {dt:.1f}s")
        return self.pipeline

    def test(self, setting, test=0):
        from utils.tools import visual
        from utils.metrics import metric

        test_data, _ = self._get_data(flag="test")

        seq_len = int(getattr(self.args, "seq_len"))
        pred_len = int(getattr(self.args, "pred_len"))
        eval_batch = int(getattr(self.args, "chronos2_eval_batch_size", 32))  # 这里建议别太大，避免显存压力

        # ========== folders align with VLM ==========
        test_vis_folder = "./test_results/" + setting + "/"
        os.makedirs(test_vis_folder, exist_ok=True)

        folder_path = "./results/" + setting + "/"
        os.makedirs(folder_path, exist_ok=True)

        # ========== prepare data ==========
        x = test_data.data_x
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)

        # Ensure (T, C): row=time step, col=feature/var  —— 和 VLM 一致
        if x.ndim == 1:
            x = x[:, None]
        T, C = x.shape

        # VLM 的 dataset.__len__ = len(data_x) - seq_len - pred_len + 1
        max_i = T - seq_len - pred_len + 1
        if max_i <= 0:
            raise ValueError(f"Test split too short: T={T}, seq_len={seq_len}, pred_len={pred_len}")

        f_dim = -1 if getattr(self.args, "features", "") == "MS" else 0

        preds_list = []
        trues_list = []

        # ========== sliding windows ==========
        # 为了效率：按 eval_batch 把多个窗口打包，一次 predict_quantiles
        with torch.no_grad():
            self.pipeline.model.eval() if hasattr(self.pipeline, "model") else None  # best-effort

            start = 0
            batch_idx = 0
            while start < max_i:
                end = min(start + eval_batch, max_i)
                B = end - start

                # Build batch contexts: (B(number of tasks!), C, seq_len), B is group size, different from chronos2's batch_size
                ctx_batch = np.empty((B, C, seq_len), dtype=np.float32)
                true_batch = np.empty((B, pred_len, C), dtype=np.float32) # also known as (b, h, c)

                for j, i in enumerate(range(start, end)):
                    ctx = x[i : i + seq_len]                              # (seq_len, C)
                    fut = x[i + seq_len : i + seq_len + pred_len]         # (pred_len, C)

                    # ctx -> (C, seq_len)
                    ctx_batch[j] = ctx.T                                  # (C, seq_len), align with chronos2 input
                    true_batch[j] = fut

                ctx_inputs = torch.from_numpy(ctx_batch)  # (B(number of tasks!), C, seq_len)
                ctx_vis = np.transpose(ctx_batch, (0, 2, 1)).copy()  # (B, seq_len, C)

                # ===== predict median (q=0.5) =====
                # mean: list[tensor], each tensor corresponds to one input series in the batch
                # mean: (task_indices(B), target_indx_inner(C), h(pred_len))
                _, mean = self.pipeline.predict_quantiles( 
                    inputs=ctx_inputs,
                    prediction_length=pred_len,
                    quantile_levels=[0.5],
                    # batch_size=B*C,              # total TS number for each Chronos2 predict batch, could be B*C here, but default is 256.
                    context_length=seq_len,
                )

                # mean is list length B; each element shape often (C, pred_len)
                pred_batch = np.empty((B, pred_len, C), dtype=np.float32)
                for j in range(B):
                    p = mean[j].detach().cpu().numpy() # p is (target_indx_inner(C), h(pred_len))
                    if p.ndim != 2:
                        raise ValueError(...)
                    if p.shape[1] != pred_len:
                        raise ValueError(...)
                    p = p.T  # (pred_len, C)
                    pred_batch[j] = p  # so pred_batch is (task_indices(B),  h(pred_len), target_indx_inner(C)), vlm's outputs format

                # --- inverse transform (exactly like VLM) ---
                if getattr(test_data, "scale", False) and getattr(self.args, "inverse", False):
                    shape = pred_batch.shape  # (B, pred_len, C)
                    pred_batch = test_data.inverse_transform(pred_batch.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    true_batch = test_data.inverse_transform(true_batch.reshape(shape[0] * shape[1], -1)).reshape(shape)

                    # for visual, inverse the context too (like VLM does for input)
                    ctx_shape = ctx_vis.shape  # (B, seq_len, C)
                    ctx_vis = test_data.inverse_transform(ctx_vis.reshape(ctx_shape[0] * ctx_shape[1], -1)).reshape(ctx_shape)

                # --- feature slicing (exactly like VLM) ---
                pred_batch = pred_batch[:, :, f_dim:]
                true_batch = true_batch[:, :, f_dim:]
                ctx_vis = ctx_vis[:, :, f_dim:]

                preds_list.append(pred_batch)
                trues_list.append(true_batch)

                # # --- visual (align with VLM's i%20==0) ---
                # if batch_idx % 20 == 0:
                #     gt = np.concatenate((ctx_vis[0, :, -1], true_batch[0, :, -1]), axis=0)
                #     pd_ = np.concatenate((ctx_vis[0, :, -1], pred_batch[0, :, -1]), axis=0)
                #     visual(gt, pd_, os.path.join(test_vis_folder, str(batch_idx) + ".pdf"))

                start = end
                batch_idx += 1

        # ========== concat + reshape exactly like VLM ==========
        preds = np.concatenate(preds_list, axis=0)   # (N, pred_len, C')
        trues = np.concatenate(trues_list, axis=0)   # (N, pred_len, C')
        print("test shape:", preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # ========== metrics + save exactly like VLM ==========
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        dtw = "not calculated"
        print(f"mse: {mse}, mae: {mae}, dtw: {dtw}")

        with open("result_long_term_forecast.txt", "a", encoding="utf-8") as f:
            f.write(setting + "  \n")
            f.write(f"mse: {mse}, mae: {mae}, dtw: {dtw}")
            f.write("\n\n")

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)

        return
