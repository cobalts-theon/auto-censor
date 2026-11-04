from unittest.mock import Mock
import cv2
import pytest
from privacy_vision.desktop import camera


def test_requested_exposure_raw_v4l_and_report(monkeypatch):
    cap=Mock(); cap.isOpened.return_value=True; cap.set.return_value=True
    cap.get.return_value=30.
    monkeypatch.setattr(camera.cv2,'VideoCapture',Mock(return_value=cap))
    configured=camera.open_camera(1,camera.CameraOptions(30,True,160))
    assert 'driver 30' in configured.info
    calls=cap.set.call_args_list
    mode=calls.index(((cv2.CAP_PROP_MODE,0),{}))
    auto=calls.index(((cv2.CAP_PROP_AUTO_EXPOSURE,1),{}))
    exposure=calls.index(((cv2.CAP_PROP_EXPOSURE,160),{}))
    assert mode < auto < exposure
    configured.release(); cap.release.assert_called_once()


def test_unsupported_manual_control_reported(monkeypatch):
    cap=Mock(); cap.isOpened.return_value=True; cap.get.return_value=15.
    cap.set.side_effect=lambda prop,value: prop != cv2.CAP_PROP_MODE
    monkeypatch.setattr(camera.cv2,'VideoCapture',Mock(return_value=cap))
    result=camera.open_camera(0,camera.CameraOptions(manual_exposure=True))
    assert 'từ chối exposure' in result.info
    assert not any(call.args[0]==cv2.CAP_PROP_EXPOSURE for call in cap.set.call_args_list)


def test_camera_not_open_released(monkeypatch):
    cap=Mock(); cap.isOpened.return_value=False
    monkeypatch.setattr(camera.cv2,'VideoCapture',Mock(return_value=cap))
    with pytest.raises(RuntimeError): camera.open_camera(0,camera.CameraOptions())
    cap.release.assert_called_once()


def test_default_driver_ring_has_four_buffers(monkeypatch):
    cap=Mock(); cap.isOpened.return_value=True; cap.set.return_value=True; cap.get.return_value=4.
    monkeypatch.setattr(camera.cv2,'VideoCapture',Mock(return_value=cap))
    result=camera.open_camera(1,camera.CameraOptions())
    assert camera.CameraOptions().buffers == 4
    cap.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE,4)
    assert 'buffer driver 4' in result.info


def test_camera_diagnostic_measures_read_without_pipeline(monkeypatch):
    import numpy as np
    from privacy_vision.cli import camera_benchmark as diagnostics
    clock=[0.]
    def read():
        clock[0]+=.05
        return True,np.zeros((1,1,3),np.uint8)
    cap=Mock(); cap.read.side_effect=read
    monkeypatch.setattr(diagnostics,'perf_counter',lambda:clock[0])
    report=diagnostics.measure_camera(cap,seconds=1,warmup=2)
    assert report['fps'] == pytest.approx(20)
    assert report['read_mean_ms'] == pytest.approx(50)
    assert set(report) == {'frames','elapsed_s','fps','read_mean_ms','read_p95_ms'}


def test_camera_diagnostic_read_failure():
    from privacy_vision.cli.camera_benchmark import measure_camera
    cap=Mock(); cap.read.return_value=(False,None)
    with pytest.raises(RuntimeError): measure_camera(cap,warmup=0)


def test_manual_exposure_restores_previous_values(monkeypatch):
    cap=Mock(); cap.isOpened.return_value=True; cap.set.return_value=True
    values={cv2.CAP_PROP_AUTO_EXPOSURE:3., cv2.CAP_PROP_EXPOSURE:600.,
            cv2.CAP_PROP_FOURCC:float(cv2.VideoWriter_fourcc(*'MJPG')),
            cv2.CAP_PROP_FRAME_WIDTH:640.,cv2.CAP_PROP_FRAME_HEIGHT:480.}
    cap.get.side_effect=lambda prop:values.get(prop,30.)
    monkeypatch.setattr(camera.cv2,'VideoCapture',Mock(return_value=cap))
    result=camera.open_camera(1,camera.CameraOptions(manual_exposure=True,exposure=80))
    assert "thực nhận 'MJPG'" in result.info
    assert result.release() is True
    assert cap.set.call_args_list[-3].args == (cv2.CAP_PROP_AUTO_EXPOSURE,1)
    assert cap.set.call_args_list[-2].args == (cv2.CAP_PROP_EXPOSURE,600.)
    assert cap.set.call_args_list[-1].args == (cv2.CAP_PROP_AUTO_EXPOSURE,3.)
    cap.release.assert_called_once()


def test_diagnostic_exposure_and_format_matrix(monkeypatch,capsys):
    from privacy_vision.cli import camera_benchmark as diagnostics
    import json
    configs=[]
    def factory(index,options):
        configs.append(options)
        cap=Mock(info='no images'); cap.release.return_value=True
        return cap
    monkeypatch.setattr(camera,'open_camera',factory)
    monkeypatch.setattr(diagnostics,'measure_camera',lambda *a:{'fps':30.})
    monkeypatch.setattr('sys.argv',['benchmark_camera.py','--buffers','4','--formats','MJPG,YUYV','--compare-exposure'])
    assert diagnostics.main()==0
    reports=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(reports)==6
    assert [(c.fourcc,c.manual_exposure,c.exposure) for c in configs] == [
        ('MJPG',False,160),('MJPG',True,80),('MJPG',True,160),
        ('YUYV',False,160),('YUYV',True,80),('YUYV',True,160)]
    assert all(r['exposure_restored'] for r in reports)


def test_requested_hd_resolution(monkeypatch):
    cap=Mock(); cap.isOpened.return_value=True; cap.set.return_value=True; cap.get.return_value=30.
    monkeypatch.setattr(camera.cv2,'VideoCapture',Mock(return_value=cap))
    camera.open_camera(1,camera.CameraOptions(width=1280,height=720))
    cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT,720)
