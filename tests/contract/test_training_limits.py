import pytest

from aire.core.errors import ConfigurationError
from aire.training.quantize import QuantizationConfig, Quantizer


def test_quantization_stub_does_not_claim_changed_weights():
    quantizer = Quantizer(config=QuantizationConfig(method="stub", bits=4))
    result = quantizer.prepare()
    assert result["stub"] is True
    assert "no real quantization" in result["honesty"]


def test_missing_quantization_extra_has_actionable_error(monkeypatch):
    from aire.training import quantize

    monkeypatch.setattr(quantize.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ConfigurationError) as error:
        Quantizer().prepare()
    assert error.value.code == "training.quant_missing"
    assert error.value.context["extra"] == "aire-ai[peft]"
