"""Bittensor Synapse protocol for Proof of Mind.

Defines the wire protocol between miners and validators.
Miners return inference results + MindPrint.
Validators verify geometric integrity without re-running inference
(model-signature mode) or with a reference run (full verification mode).
"""

from typing import List, Optional

import bittensor as bt


class MindPrintSynapse(bt.Synapse):
    """Synapse for verified inference requests.

    Flow:
      1. Validator sends: query, model_id
      2. Miner runs inference, generates MindPrint
      3. Miner returns: response, mindprint_b64, model_id
      4. Validator verifies MindPrint against reference or model signature
    """

    # --- Request fields (validator → miner) ---
    query: str = ""
    model_id: str = ""
    max_tokens: int = 256
    temperature: float = 0.0  # Greedy by default for deterministic geometry
    seed: int = 42  # Fixed seed for reproducibility

    # --- Response fields (miner → validator) ---
    response: str = ""
    mindprint_b64: str = ""  # Base64-encoded compact MindPrint
    n_tokens: Optional[int] = None
    extraction_time_ms: Optional[float] = None

    # --- Verification metadata ---
    # Set by validator after verification
    verification_passed: Optional[bool] = None
    verification_confidence: Optional[float] = None

    def deserialize(self) -> dict:
        """Deserialize response for downstream use."""
        return {
            "response": self.response,
            "mindprint_b64": self.mindprint_b64,
            "model_id": self.model_id,
            "n_tokens": self.n_tokens,
        }
