import cv2
import numpy as np
from privacy_vision.schemas import Detection
from privacy_vision.processing.tracking import RegionTracker


def detection(box=(35,35,105,105),source='face',name='face'):
    return Detection(source,0,name,.9,box)


def test_hold_expiry_and_current_detection_coverage():
    frame=np.zeros((160,200,3),np.uint8)
    t=RegionTracker()
    t.update(frame,[detection()],0)
    held=t.update(frame,[],.2)
    assert held[0].metadata['tracking_state']=='held'
    result=t.update(frame,[detection((40,35,110,105))],.3)
    assert len(result)==1
    assert result[0].bbox[0]<=40 and result[0].bbox[2]>=110
    assert result[0].metadata['tracking_state']=='detected'
    assert t.update(frame,[],1.)==[]


def test_flow_follows_moving_face():
    frame=np.zeros((160,200,3),np.uint8)
    frame[40:100,40:100]=np.random.default_rng(4).integers(0,256,(60,60,3),dtype=np.uint8)
    t=RegionTracker(); t.update(frame,[detection()],0)
    moved=cv2.warpAffine(frame,np.float32([[1,0,5],[0,1,3]]),(200,160))
    result=t.update(moved,[],.1)
    assert result[0].metadata['tracking_state']=='tracked'
    assert abs(result[0].bbox[0]-40)<=1 and abs(result[0].bbox[1]-38)<=1


def test_separate_objects_and_classes_not_matched():
    frame=np.zeros((160,300,3),np.uint8)
    t=RegionTracker()
    ds=[detection((10,10,50,50)),detection((200,10,240,50))]
    t.update(frame,ds,0)
    assert len(t.update(frame,list(reversed(ds)),.1))==2
    t=RegionTracker(); t.update(frame,[detection()],0)
    result=t.update(frame,[detection(source='plate',name='license_plate')],.1)
    assert len(result)==2


def test_resolution_change_clears_old_tracks():
    t=RegionTracker()
    t.update(np.zeros((720,1280,3),np.uint8),[detection()],0)
    assert t.update(np.zeros((1080,1920,3),np.uint8),[],.1)==[]


def test_document_requires_confirmation_then_nested_box_keeps_one_track():
    frame=np.zeros((200,300,3),np.uint8)
    t=RegionTracker()
    full=detection((30,30,180,130),'card','document_region')
    assert t.update(frame,[full],0)==[]
    result=t.update(frame,[full],.04)
    assert len(result)==1
    partial=detection((60,50,120,80),'card','document_region')
    result=t.update(frame,[partial],.08)
    assert len(result)==1 and len(t.tracks)==1
    assert result[0].bbox[0]<40 and result[0].bbox[2]>160
    held=t.update(frame,[],.12)
    assert len(held)==1 and held[0].metadata['tracking_state']=='held'
    assert t.update(frame,[],.4)==[]


def test_one_frame_document_outlier_never_renders():
    frame=np.zeros((200,300,3),np.uint8)
    t=RegionTracker()
    assert t.update(frame,[detection((30,30,180,130),'card','document_region')],0)==[]
    assert t.update(frame,[],.04)==[]


def test_two_confirmed_documents_remain_independent():
    frame=np.zeros((200,500,3),np.uint8)
    t=RegionTracker()
    ds=[detection((20,30,170,130),'card','document_region'),
        detection((300,30,450,130),'card','document_region')]
    t.update(frame,ds,0)
    result=t.update(frame,list(reversed(ds)),.04)
    assert len(result)==2
    assert result[0].bbox[2]<result[1].bbox[0]


def test_moving_document_does_not_accumulate_a_trailing_rectangle():
    frame=np.zeros((400,1200,3),np.uint8)
    tracker=RegionTracker()
    widths=[]
    for i in range(100):
        x=40+4*i
        result=tracker.update(frame,[detection((x,80,x+160,180),'card','document_region')],i/30)
        if result:
            box=result[0].bbox
            widths.append(box[2]-box[0])
            assert box[0]<=x and box[2]>=x+160
            assert x-box[0] <= 3
    assert max(widths)<=163


def test_filtered_box_converges_after_outward_jitter():
    frame=np.zeros((300,500,3),np.uint8)
    tracker=RegionTracker()
    tracker.update(frame,[detection((40,40,210,160),'card','document_region')],0)
    for i in range(1,100):
        result=tracker.update(frame,[detection((50,50,200,150),'card','document_region')],i/30)
    box=result[0].bbox
    assert box[0]>=49 and box[1]>=49 and box[2]<=201 and box[3]<=151


def test_nested_partial_proposals_do_not_pull_card_toward_a_corner():
    frame=np.zeros((300,500,3),np.uint8)
    tracker=RegionTracker()
    full=detection((30,30,230,180),'card','document_region')
    tracker.update(frame,[full],0)
    tracker.update(frame,[full],.03)
    for i in range(2,7):
        result=tracker.update(frame,[detection((35,35,90,65),'card','document_region')],i/30)
        assert result[0].bbox==(30,30,230,180)
    assert tracker.update(frame,[],.4)==[]


def test_nonconsecutive_document_proposals_do_not_confirm():
    frame=np.zeros((200,300,3),np.uint8)
    tracker=RegionTracker()
    doc=detection((20,20,180,120),'card','document_region')
    assert tracker.update(frame,[doc],0)==[]
    assert tracker.update(frame,[],.03)==[]
    assert tracker.update(frame,[doc],.06)==[]


def test_sudden_large_proposal_does_not_instantly_expand_confirmed_card():
    frame=np.zeros((500,800,3),np.uint8)
    tracker=RegionTracker()
    small=detection((100,100,260,200),'card','document_region')
    large=detection((50,50,600,400),'card','document_region')
    tracker.update(frame,[small],0)
    tracker.update(frame,[small],.03)
    result=tracker.update(frame,[large],.06)
    assert len(result)==1 and result[0].bbox==small.bbox
    result=tracker.update(frame,[small],.09)
    assert len(result)==1 and result[0].bbox==small.bbox


def test_confirmed_large_replacement_retires_old_nested_hypothesis():
    frame=np.zeros((500,800,3),np.uint8)
    tracker=RegionTracker()
    small=detection((100,100,260,200),'card','document_region')
    large=detection((50,50,600,400),'card','document_region')
    tracker.update(frame,[small],0)
    tracker.update(frame,[small],.03)
    tracker.update(frame,[large],.06)
    result=tracker.update(frame,[large],.09)
    assert len(result)==1 and result[0].bbox==large.bbox


def test_document_continuous_tilt_tracking():
    frame = np.zeros((300, 300, 3), np.uint8)
    frame[50:250, 50:200] = np.random.default_rng(42).integers(50, 250, (200, 150, 3), dtype=np.uint8)
    t = RegionTracker(hold_seconds=0.6, document_hold_seconds=0.6)
    doc = Detection('document', 0, 'document_region', 0.9, (50, 50, 200, 250))
    t.update(frame, [doc], 0)

    current_frame = frame
    # Rotate 2 deg each frame for 5 frames without detector
    for step in range(1, 6):
        M = cv2.getRotationMatrix2D((125, 150), 2.0, 1.0)
        current_frame = cv2.warpAffine(current_frame, M, (300, 300))
        res = t.update(current_frame, [], step * 0.033)
        assert len(res) == 1
        assert res[0].metadata['tracking_state'] == 'tracked'

