from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
from privacy_vision.processing.boxes import normalize_box
from privacy_vision.config import DetectorConfig, PipelineConfig
from privacy_vision.schemas import Detection
from privacy_vision.pipeline import PrivacyPipeline
from privacy_vision.image_io import read_image, write_image
from privacy_vision.detectors.yunet import YuNetDetector
from privacy_vision.detectors.yolo import YoloDetector

@pytest.mark.parametrize('box,expected', [((-2,-3,15,12),(0,0,10,10)), ((2,2,2,5),None), ((5,5,2,8),None), ((11,1,20,8),None), ((float('nan'),0,1,1),None)])
def test_boxes(box, expected):
    assert normalize_box(box,10,10) == expected

def test_margin():
    assert normalize_box((2,2,8,8),10,10,.5) == (0,0,10,10)

def make_pipeline(mode='black', source='face', rows=None):
    d = Mock(source=source)
    d.detect.return_value = rows or []
    c = PipelineConfig([DetectorConfig(source, Path('unused'))], mode=mode, margin=0)
    return PrivacyPipeline(c, [d]), d

@pytest.mark.parametrize('mode', ['black','blur','pixelate'])
def test_redaction(mode):
    im = np.random.default_rng(5).integers(0,256,(20,20,3),dtype=np.uint8)
    rows = [Detection('face',0,'face',.9,(-2,-1,8,8)), Detection('face',0,'face',.8,(4,4,12,12))]
    p,_ = make_pipeline(mode, rows=rows)
    r = p.process(im)
    np.testing.assert_array_equal(r.image[12:], im[12:])
    assert not np.array_equal(r.image[:8,:8], im[:8,:8])
    if mode == 'black':
        assert not r.image[:8,:8].any()
        assert not r.image[4:12,4:12].any()
    assert r.detections[0].bbox == (0,0,8,8)
    assert im.any()

@pytest.mark.parametrize('source',['face','plate','card','cccd'])
def test_enabled_empty_tiny(source):
    p,d = make_pipeline(source=source)
    im = np.full((1,1,3),100,np.uint8)
    np.testing.assert_array_equal(p.process(im).image, im)
    d.detect.assert_called_once()
    for mode in ['black','blur','pixelate']:
        p,_ = make_pipeline(mode, source, [Detection(source,0,'field',.9,(0,0,1,1))])
        assert p.process(im).image.shape == im.shape

def test_disabled():
    p = PrivacyPipeline(PipelineConfig([]), [])
    assert not p.process(np.zeros((2,2,3),np.uint8)).detections

@pytest.mark.parametrize('kind',[YuNetDetector,YoloDetector])
def test_missing(kind,tmp_path):
    c = DetectorConfig('face',tmp_path/'missing.pt')
    with pytest.raises(FileNotFoundError):
        kind(c) if kind is YuNetDetector else kind(c,'cpu')

def test_failure_no_output(tmp_path):
    p,d = make_pipeline()
    d.detect.side_effect = RuntimeError('failure')
    src,out = tmp_path/'ảnh.png',tmp_path/'out.png'
    write_image(src,np.zeros((10,10,3),np.uint8))
    with pytest.raises(RuntimeError):
        p.process_file(src,out)
    assert not out.exists()
    out.write_bytes(b'previous')
    with pytest.raises(RuntimeError):
        p.process_file(src,out)
    assert out.read_bytes() == b'previous'

def test_invalid_and_unicode(tmp_path):
    p,_ = make_pipeline()
    with pytest.raises(ValueError): p.process(np.zeros((0,0,3),np.uint8))
    bad=tmp_path/'bad.png'; bad.write_bytes(b'bad')
    with pytest.raises(ValueError): read_image(bad)
    path=tmp_path/'tiếng Việt.png'
    im=np.full((5,5,3),123,np.uint8)
    write_image(path,im)
    np.testing.assert_array_equal(read_image(path),im)

def test_adapter_normalization():
    from privacy_vision.detectors.yolo import YoloDetector
    d=YoloDetector.__new__(YoloDetector)
    d.source='plate'; d.device='cpu'; d.config=DetectorConfig('plate',Path('unused'))
    result=Mock(); result.names={0:'license_plate'}
    result.boxes.data.cpu.return_value.tolist.return_value=[[1,2,8,9,.9,0]]
    d.model=Mock(); d.model.predict.return_value=[result]
    row=d.detect(np.zeros((10,10,3),np.uint8))[0]
    assert row == Detection('plate',0,'license_plate',.9,(1,2,8,9))
    assert d.model.predict.call_args.kwargs['half'] is False

def test_cli_missing_input_before_model_load(tmp_path, monkeypatch, capsys):
    from privacy_vision.cli import image as cli
    factory = Mock()
    monkeypatch.setattr(cli, 'PrivacyPipeline', factory)
    monkeypatch.setattr('sys.argv', ['predict_image.py', '--input', str(tmp_path/'missing.jpg'),
                                   '--output', str(tmp_path/'out.png')])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert 'Không tìm thấy file ảnh --input' in capsys.readouterr().err
    factory.assert_not_called()
    assert not (tmp_path/'out.png').exists()

@pytest.mark.parametrize('angle,box', [(0,(2,1,7,4)), (90,(1,3,4,8)), (180,(3,2,8,5)), (270,(2,2,5,7))])
def test_unrotate_non_square(angle, box):
    from privacy_vision.processing.boxes import unrotate_box
    assert unrotate_box(box,10,6,angle) == (2,1,7,4)

def test_document_rotation_pipeline():
    c=PipelineConfig([DetectorConfig('card',Path('unused'))],margin=0,
                     document_rotations=(0,90))
    d=Mock(source='card')
    d.detect.side_effect=[[],[Detection('card',6,'id',.9,(1,3,4,8))]]
    p=PrivacyPipeline(c,[d])
    im=np.full((6,10,3),255,np.uint8)
    result=p.process(im)
    assert result.detections[0].bbox == (2,1,7,4)
    assert result.detections[0].metadata['rotation_degrees'] == 90
    assert not result.image[1:4,2:7].any()
    assert d.detect.call_args_list[1].args[0].shape == (10,6,3)

def test_rotation_failure_no_output(tmp_path):
    c=PipelineConfig([DetectorConfig('cccd',Path('unused'))], document_rotations=(0,90))
    d=Mock(source='cccd'); d.detect.side_effect=[[],RuntimeError('rotation failed')]
    p=PrivacyPipeline(c,[d]); src=tmp_path/'src.png'; out=tmp_path/'out.png'
    write_image(src,np.zeros((6,10,3),np.uint8))
    with pytest.raises(RuntimeError): p.process_file(src,out)
    assert not out.exists()


def test_pipeline_tracking_masks_dropped_detection():
    from privacy_vision.processing.tracking import RegionTracker
    p,d=make_pipeline(rows=[Detection('face',0,'face',.9,(2,2,8,8))])
    tracker=RegionTracker()
    image=np.full((10,10,3),200,np.uint8)
    first=p.process(image,tracker=tracker)
    d.detect.return_value=[]
    second=p.process(image,tracker=tracker)
    assert not first.image[2:8,2:8].any()
    assert not second.image[2:8,2:8].any()
    assert second.detections[0].metadata['tracking_state']=='held'
    assert 'tracking' in second.timings_ms
