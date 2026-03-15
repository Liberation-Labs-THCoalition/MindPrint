"""MindPrint serialization — compact binary encoding for network transport.

Target: ~500 bytes for a 32-layer model with stride-4 sampling (8 layers).

Wire format (v1):
  [1B version] [32B content_hash] [4x8B aggregates] [2B n_layers]
  [2B n_extracted] [4B n_tokens] [n_extracted * 4B rank_profile]
  [n_extracted * 4B entropy_profile] [32B fingerprint_hex_prefix]

All floats are IEEE 754 float32 (4 bytes). Big-endian.
"""

import struct
from typing import Optional

from mindprint.proof.mindprint import MindPrint


# Header: version(1) + content_hash(16 raw) + 4 aggregates(4*4=16) + n_layers(2) + n_extracted(2) + n_tokens(4)
_HEADER_FMT = ">B 16s ffff HH I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 1 + 16 + 16 + 2 + 2 + 4 = 41 bytes

# Fingerprint suffix: 16 bytes (first 32 hex chars = 16 raw bytes)
_FINGERPRINT_SIZE = 16

# Per-layer: 2 floats (rank + entropy) = 8 bytes each
_PER_LAYER_FMT = ">ff"
_PER_LAYER_SIZE = struct.calcsize(_PER_LAYER_FMT)  # 8 bytes


def encode_mindprint(mp: MindPrint) -> bytes:
    """Encode a MindPrint to compact binary format.

    Returns bytes. Typical size for 8 extracted layers: 41 + 64 + 16 = 121 bytes.
    For 32 layers: 41 + 256 + 16 = 313 bytes.
    """
    # Pack header
    content_hash_raw = bytes.fromhex(mp.content_hash[:32])
    header = struct.pack(
        _HEADER_FMT,
        mp.version,
        content_hash_raw,
        mp.norm,
        mp.norm_per_token,
        mp.key_rank,
        mp.key_entropy,
        mp.n_layers,
        mp.n_extracted,
        mp.n_tokens or 0,
    )

    # Pack per-layer profiles
    layers = b""
    for rank, entropy in zip(mp.layer_rank_profile, mp.layer_entropy_profile):
        layers += struct.pack(_PER_LAYER_FMT, rank, entropy)

    # Pack fingerprint prefix (first 32 hex chars = 16 bytes)
    fp_raw = bytes.fromhex(mp.fingerprint[:32]) if mp.fingerprint else b"\x00" * 16

    return header + layers + fp_raw


def decode_mindprint(data: bytes, model_id: str = "") -> MindPrint:
    """Decode a MindPrint from compact binary format.

    Args:
        data: Raw bytes from encode_mindprint().
        model_id: Model identifier (not stored in binary, must be provided).

    Returns:
        MindPrint (fingerprint will be the 32-char hex prefix only).
    """
    # Unpack header
    (
        version,
        content_hash_raw,
        norm,
        norm_per_token,
        key_rank,
        key_entropy,
        n_layers,
        n_extracted,
        n_tokens_raw,
    ) = struct.unpack(_HEADER_FMT, data[:_HEADER_SIZE])

    content_hash = content_hash_raw.hex()
    n_tokens = n_tokens_raw if n_tokens_raw > 0 else None

    # Unpack per-layer profiles
    layer_data_start = _HEADER_SIZE
    layer_data_end = layer_data_start + n_extracted * _PER_LAYER_SIZE

    rank_profile = []
    entropy_profile = []
    offset = layer_data_start
    for _ in range(n_extracted):
        rank, entropy = struct.unpack(_PER_LAYER_FMT, data[offset : offset + _PER_LAYER_SIZE])
        rank_profile.append(rank)
        entropy_profile.append(entropy)
        offset += _PER_LAYER_SIZE

    # Unpack fingerprint prefix
    fp_raw = data[layer_data_end : layer_data_end + _FINGERPRINT_SIZE]
    fingerprint = fp_raw.hex() if len(fp_raw) == _FINGERPRINT_SIZE else ""

    return MindPrint(
        version=version,
        model_id=model_id,
        content_hash=content_hash,
        norm=norm,
        norm_per_token=norm_per_token,
        key_rank=key_rank,
        key_entropy=key_entropy,
        layer_rank_profile=rank_profile,
        layer_entropy_profile=entropy_profile,
        n_layers=n_layers,
        n_extracted=n_extracted,
        n_tokens=n_tokens,
        fingerprint=fingerprint,
    )


def mindprint_size(n_extracted: int) -> int:
    """Calculate wire size in bytes for a given number of extracted layers."""
    return _HEADER_SIZE + n_extracted * _PER_LAYER_SIZE + _FINGERPRINT_SIZE
