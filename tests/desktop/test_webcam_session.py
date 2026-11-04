from unittest.mock import Mock
import numpy as np
import pytest
from privacy_vision.config import PipelineConfig
from privacy_vision.desktop.session import PreviewBuffer, Preview, run_session
from privacy_vision.schemas import PipelineResult


def test_bounded_mailbox():
    buffer=PreviewBuffer()
    first=Preview(np.zeros((2,2,3),np.uint8),1,2,{},'cpu')
    last=Preview(np.ones((2,2,3),np.uint8),1,2,{},'cpu')
    buffer.publish(first); buffer.publish(last)
    assert buffer.take()[0] is last
    assert buffer.take()[0] is None
    buffer.publish(first); buffer.clear('Lỗi')
    assert buffer.take() == (None,'Lỗi')


@pytest.mark.parametrize('failure',['read','inference','load'])
def test_failure_clears_preview_and_releases(failure, caplog):
    raw=np.full((4,4,3),255,np.uint8)
    cap=Mock(); cap.read.return_value=(True,raw)
    pipeline=Mock(); pipeline.process.side_effect=RuntimeError('private error')
    if failure == 'read': cap.read.return_value=(False,None)
    factory=Mock(return_value=pipeline)
    if failure == 'load': factory.side_effect=RuntimeError('load error')
    buffer=PreviewBuffer(); buffer.publish(Preview(raw,0,0,{},'cpu'))
    run_session(PipelineConfig(),0,buffer,lambda:False,factory,lambda _:cap,lambda _:'cpu')
    image,status=buffer.take()
    assert image is None and status.startswith('Lỗi')
    assert {'read': 'camera', 'inference': 'AI/redaction', 'load': 'tải model'}[failure] in status
    assert 'private error' not in caplog.text and 'load error' not in caplog.text
    if failure != 'load': cap.release.assert_called_once()
    if failure == 'read': pipeline.process.assert_not_called()


def test_only_redacted_frame_published_and_stop_releases():
    raw=np.full((4,4,3),255,np.uint8)
    safe=np.zeros_like(raw)
    pipeline=Mock(); pipeline.process.return_value=PipelineResult(safe,[],{'total':2})
    cap=Mock(); cap.read.return_value=(True,raw)
    buffer=PreviewBuffer(); published=[]
    original=buffer.publish
    def publish(item):
        published.append(item); original(item)
    buffer.publish=publish
    run_session(PipelineConfig(),0,buffer,lambda:bool(published),lambda _:pipeline,
                lambda _:cap,lambda _:'CPU')
    assert len(published)==1
    assert not published[0].image.any()
    assert not np.shares_memory(published[0].image,safe)
    cap.release.assert_called_once()
    assert buffer.take() == (None,'Đã dừng')


def test_no_detectors_does_not_open_camera():
    capture=Mock(); buffer=PreviewBuffer()
    run_session(PipelineConfig([]),0,buffer,lambda:False,capture_factory=capture)
    capture.assert_not_called()
    assert buffer.take()[1].startswith('Lỗi')

@pytest.mark.parametrize('output', [None, np.zeros((1,1),np.float32), np.zeros((2,2,3),np.uint8)])
def test_invalid_pipeline_result_never_published(output):
    cap=Mock(); cap.read.return_value=(True,np.full((4,4,3),255,np.uint8))
    p=Mock(); p.process.return_value=PipelineResult(output,[],{'total':1})
    b=PreviewBuffer(); b.publish=Mock()
    run_session(PipelineConfig(),0,b,lambda:False,lambda _:p,lambda _:cap,lambda _:'cpu')
    b.publish.assert_not_called()
    assert b.take()[1].startswith('Lỗi')


def test_raw_storage_returned_by_pipeline_is_rejected():
    raw=np.full((4,4,3),255,np.uint8)
    cap=Mock(); cap.read.return_value=(True,raw)
    p=Mock(); p.process.return_value=PipelineResult(raw,[],{'total':1})
    b=PreviewBuffer(); b.publish=Mock()
    run_session(PipelineConfig(),0,b,lambda:False,lambda _:p,lambda _:cap,lambda _:'cpu')
    b.publish.assert_not_called()
    assert b.take()[1].startswith('Lỗi')


def test_document_strategy_is_included_in_device_label():
    from privacy_vision.desktop.session import device_label
    p=Mock(device='cpu',config=Mock(detectors=[Mock(source='face')],document_detection_strategy='detector'))
    assert device_label(p)=='CPU (CUDA không dùng)'
