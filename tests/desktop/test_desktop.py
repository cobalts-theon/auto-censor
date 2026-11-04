"""Offscreen Qt lifecycle tests; no camera, model, GPU or display required."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import threading
import time
from unittest.mock import Mock
import pytest
pytest.importorskip('PySide6')
from PySide6.QtWidgets import QApplication
from privacy_vision.desktop import window

@pytest.fixture(scope='module')
def qapp():
    return QApplication.instance() or QApplication([])


def test_empty_selection_no_worker(qapp,tmp_path):
    w=window.WebcamWindow(tmp_path)
    for check in w.checks.values(): check.setChecked(False)
    w.start_session()
    assert w.worker is None
    assert 'ít nhất' in w.status.text()
    w.close()


def test_worker_load_does_not_block_gui_and_closes_cooperatively(qapp,tmp_path,monkeypatch):
    entered=threading.Event(); release=threading.Event()
    def session(config,camera,buffer,stopped,**kwargs):
        entered.set()
        release.wait(3)
        buffer.clear('Đã dừng')
    monkeypatch.setattr(window,'run_session',session)
    w=window.WebcamWindow(tmp_path)
    w.start_session()
    assert entered.wait(2)
    qapp.processEvents()
    assert w.worker.isRunning()
    assert not w.start_button.isEnabled()
    w.close()
    assert w.closing
    assert w.worker.isInterruptionRequested()
    release.set()
    deadline=time.monotonic()+3
    while w.worker is not None and time.monotonic()<deadline:
        qapp.processEvents()
        time.sleep(.005)
    assert w.worker is None
    w.close()


def test_error_blanks_previous_preview(qapp,tmp_path):
    import numpy as np
    from privacy_vision.desktop.session import Preview
    w=window.WebcamWindow(tmp_path)
    w.worker=Mock(); w.worker.isInterruptionRequested.return_value=False
    w.worker.session_id='test'
    w.buffer.publish(Preview(np.zeros((5,5,3),np.uint8),2,5,{'total':5},'CPU', published_at=time.perf_counter(), session_id='test'))
    w.refresh()
    assert not w.preview.pixmap().isNull()
    w.buffer.clear('Lỗi AI')
    w.refresh()
    assert w.preview.pixmap().isNull()
    assert w.status.text() == 'Lỗi AI'
    w.worker=None
    w.close()


def test_document_quality_settings_do_not_upscale_plate(qapp,tmp_path,monkeypatch):
    factory=Mock()
    monkeypatch.setattr(window,'WebcamWorker',factory)
    w=window.WebcamWindow(tmp_path)
    w.inference_size.setCurrentText('960')
    w.document_confidence.setValue(.2)
    w.start_session()
    args=factory.call_args.args
    config=args[0]
    detectors={c.source:c for c in config.detectors}
    assert detectors['card'].imgsz==960 and detectors['cccd'].imgsz==960
    assert detectors['plate'].imgsz==640 and detectors['face'].imgsz==640
    assert detectors['card'].confidence==.2 and detectors['plate'].confidence==.25
    assert args[4].width==1280 and args[4].height==720
    w.worker=None
    w.close()

@pytest.mark.parametrize('session,age',[('previous',0),('current',3)])
def test_stale_or_wrong_session_preview_blocked(qapp,tmp_path,session,age):
    import numpy as np
    from privacy_vision.desktop.session import Preview
    w=window.WebcamWindow(tmp_path)
    w.worker=Mock(session_id='current'); w.worker.isInterruptionRequested.return_value=False
    w.buffer.publish(Preview(np.zeros((5,5,3),np.uint8),1,1,{},'cpu',
                             published_at=time.perf_counter()-age,session_id=session))
    w.refresh()
    assert w.preview.pixmap().isNull()
    w.worker=None; w.close()


def test_new_document_controls_and_defaults(qapp,tmp_path):
    w=window.WebcamWindow(tmp_path)
    assert w.document_mode.currentText()=='Full document'
    assert not w.strict.isChecked()
    assert not w.strategy.model().item(1).isEnabled()
    assert not w.strategy.model().item(2).isEnabled()
    w.document_mode.setCurrentIndex(0)
    assert not w.whole_documents.isChecked()
    w.close()


def test_coverage_status_persists_and_idle_preview_expires(qapp,tmp_path):
    import numpy as np
    from privacy_vision.desktop.session import Preview
    w=window.WebcamWindow(tmp_path)
    w.worker=Mock(session_id='current'); w.worker.isInterruptionRequested.return_value=False
    w.buffer.publish(Preview(np.zeros((5,5,3),np.uint8),1,1,{},'cpu',
                             published_at=time.perf_counter(),session_id='current'))
    w.refresh()
    assert 'Coverage unverified' in w.status.text()
    w.refresh()
    assert 'Coverage unverified' in w.status.text()
    w.last_preview_at=time.perf_counter()-3
    w.refresh()
    assert w.preview.pixmap().isNull()
    w.worker=None; w.close()
