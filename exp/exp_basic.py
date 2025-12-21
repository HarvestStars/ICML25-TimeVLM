import os
import torch

from models import (
    Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer,
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer,
    FiLM, iTransformer, Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple,
    TemporalFusionTransformer, SCINet, PAttn, TimeXer, TimeLLM, VisionTS
)


class Exp_Basic(object):
    """
    Base experiment class.

    Assumptions in the original repo:
    - self._build_model() returns a torch.nn.Module (has .to and .parameters)

    New requirement:
    - self._build_model() might return a wrapper/pipeline (e.g., Chronos2Pipeline)
      which may NOT have .to() or .parameters(), but may expose an underlying
      torch model at `.model`.

    This base class now:
    - moves to device only when possible
    - counts parameters only when possible
    """

    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimeLLM': TimeLLM,
            'VisionTS': VisionTS,
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Stationary': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            'SCINet': SCINet,
            'PAttn': PAttn,
            'TimeXer': TimeXer
        }

        # Keep your dynamic TimeVLM import behavior
        if getattr(args, "model", None) == 'TimeVLM':
            from src.TimeVLM import model as TimeVLM
            self.model_dict['TimeVLM'] = TimeVLM

        self.device = self._acquire_device()
        self.model = self._build_model()

        # Best-effort device placement:
        # 1) If it's a torch module, do model.to(device)
        # 2) If it's a pipeline with .model being torch module, do pipeline.model.to(device)
        self._move_model_to_device()

        if getattr(args, "is_training", 0):
            self._log_model_parameters()

    # -------------------------
    # Helper: unwrap torch model
    # -------------------------
    @staticmethod
    def _unwrap_torch_model(obj):
        """
        Return a torch-like model (has .parameters) if possible.
        - If obj itself has .parameters -> return obj
        - Else if obj.model has .parameters -> return obj.model
        - Else None
        """
        if obj is None:
            return None
        if hasattr(obj, "parameters"):
            return obj
        if hasattr(obj, "model") and hasattr(obj.model, "parameters"):
            return obj.model
        return None

    # -------------------------
    # Device placement
    # -------------------------
    def _move_model_to_device(self):
        """
        Move model/pipeline to target device in a safe way.
        """
        # Case 1: torch.nn.Module-like
        if hasattr(self.model, "to"):
            try:
                self.model = self.model.to(self.device)
                return
            except Exception as e:
                print(f"[Warn] model.to(device) failed for {type(self.model).__name__}: {e}")

        # Case 2: pipeline wrapper exposing underlying torch model
        if hasattr(self.model, "model") and hasattr(self.model.model, "to"):
            try:
                self.model.model = self.model.model.to(self.device)
            except Exception as e:
                print(f"[Warn] model.model.to(device) failed for {type(self.model).__name__}.model: {e}")

    # -------------------------
    # Parameter logging
    # -------------------------
    def _log_model_parameters(self):
        """
        打印模型参数（兼容 pipeline/wrapper）。
        """
        tm = self._unwrap_torch_model(self.model)
        if tm is None:
            print(f"[Info] Skip parameter counting: model object "
                  f"({type(self.model).__name__}) has no .parameters() "
                  f"and no .model.parameters().")
            return

        # tm is torch-like
        try:
            learnable_params = sum(p.numel() for p in tm.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in tm.parameters())
            print(f"Learnable model parameters: {learnable_params:,}")
            print(f"Total model parameters: {total_params:,}")
        except Exception as e:
            print(f"[Warn] Parameter counting failed for {type(tm).__name__}: {e}")

    # -------------------------
    # Abstract methods
    # -------------------------
    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if getattr(self.args, "use_gpu", False):
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Use GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
