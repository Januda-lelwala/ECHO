"""Backend application package initialization."""

import os


# Some PyTorch operations used by Transformers/Captum are not implemented by
# the MPS backend. This must be configured before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
