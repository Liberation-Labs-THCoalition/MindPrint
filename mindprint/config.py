"""MindPrint configuration."""

from typing import List, Optional
from pydantic_settings import BaseSettings


class MindPrintConfig(BaseSettings):
    """Runtime configuration for MindPrint extraction."""

    # Layer sampling
    layer_subset: Optional[List[int]] = None
    layer_stride: int = 1

    # SVD
    variance_threshold: float = 0.9

    # Paths to sibling repos (for GPU feature extraction)
    cricket_src_path: Optional[str] = None
    kv_experiments_code_path: Optional[str] = None

    model_config = {"env_prefix": "MINDPRINT_"}

    def resolve_layer_subset(self, n_layers: int) -> Optional[List[int]]:
        if self.layer_subset is not None:
            return [i for i in self.layer_subset if i < n_layers]
        if self.layer_stride > 1:
            return list(range(0, n_layers, self.layer_stride))
        return None


# Alias for backward compatibility with code ported from CacheScope
CacheScopeConfig = MindPrintConfig
