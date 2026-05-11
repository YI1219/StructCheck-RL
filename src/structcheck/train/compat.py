from transformers import modeling_utils


def patch_transformers_for_minicpm() -> None:
    """Compatibility patch for MiniCPM-V remote code on newer transformers."""
    if getattr(modeling_utils.PreTrainedModel, "_structcheck_minicpm_patch", False):
        return

    original = getattr(modeling_utils.PreTrainedModel, "_adjust_tied_keys_with_tied_pointers", None)
    if original is None:
        modeling_utils.PreTrainedModel._structcheck_minicpm_patch = True
        return

    def _patched(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            tied = getattr(self, "_tied_weights_keys", None) or []
            self.all_tied_weights_keys = {k: True for k in tied}
        return original(self, *args, **kwargs)

    modeling_utils.PreTrainedModel._adjust_tied_keys_with_tied_pointers = _patched
    modeling_utils.PreTrainedModel._structcheck_minicpm_patch = True
