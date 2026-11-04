"""Opt-in real inference; user supplies their own local image."""
import os
from pathlib import Path
import pytest
from privacy_vision.config import PipelineConfig
from privacy_vision.pipeline import PrivacyPipeline
from privacy_vision.image_io import read_image

@pytest.mark.skipif(not os.environ.get('PRIVACYVISION_SMOKE_IMAGE'), reason='Set PRIVACYVISION_SMOKE_IMAGE to a local sample')
@pytest.mark.parametrize('document_mode,strict', [('fields', False), ('full', False), ('full', True)])
def test_real_models(document_mode, strict):
    config = PipelineConfig(device='cpu', document_redaction_mode=document_mode, strict_privacy_mode=strict)
    if not all(c.checkpoint.is_file() for c in config.detectors):
        pytest.skip('Missing local checkpoints')
    result = PrivacyPipeline(config).process(read_image(Path(os.environ['PRIVACYVISION_SMOKE_IMAGE'])))
    assert set(result.timings_ms) >= {'face','plate','card','cccd','redaction'}

    assert result.coverage_status == 'coverage_unverified'
    if strict:
        assert result.strict_blackout and not result.image.any()
