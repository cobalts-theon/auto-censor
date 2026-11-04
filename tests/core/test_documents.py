import numpy as np
import pytest
from privacy_vision.processing.documents import document_regions
from privacy_vision.schemas import Detection


def field(name,box,source='cccd'):
    return Detection(source,0,name,.9,box)


def fields(dx=0):
    return [field('id',(dx+50,60,dx+150,70)),field('name',(dx+50,80,dx+150,90)),
            field('dob',(dx+50,100,dx+130,110))]


def test_union_and_separate_xy_margins_without_any_border():
    image=np.zeros((300,500,3),np.uint8)
    result=document_regions(image,fields())
    assert len(result)==1
    assert result[0].class_name == 'document_region'
    assert result[0].bbox == (25,45,175,125)
    assert result[0].metadata['unique_fields']==3


def test_two_cards_not_unioned_together():
    result=document_regions(np.zeros((300,800,3),np.uint8),fields()+fields(450))
    assert len(result)==2
    assert all(d.class_name=='document_region' for d in result)
    assert result[0].bbox[2] < result[1].bbox[0]


def test_duplicate_models_not_counted_as_three_fields():
    detections=[field('id',(50,60,150,70)),field('id_',(51,60,151,70),'card'),
                field('name',(50,80,150,90))]
    result=document_regions(np.zeros((300,500,3),np.uint8),detections)
    assert all(d.class_name=='document_boundary_unresolved' for d in result)


def test_hint_needs_additional_field():
    image=np.zeros((300,500,3),np.uint8)
    title=field('cccd',(50,60,150,70),'card')
    assert document_regions(image,[title])[0].class_name=='document_boundary_unresolved'
    assert document_regions(image,[title,field('name',(50,90,150,100))])[0].class_name=='document_region'


def test_insufficient_cluster_does_not_cancel_valid_cluster():
    result=document_regions(np.zeros((300,800,3),np.uint8),fields()+[field('id',(600,60,700,70))])
    assert any(d.class_name=='document_region' for d in result)
    assert any(d.class_name=='document_boundary_unresolved' for d in result)


def test_boundary_clipping_and_no_document_detections():
    image=np.zeros((200,300,3),np.uint8)
    detections=[field(d.class_name, (0,d.bbox[1],d.bbox[2]-50,d.bbox[3])) for d in fields()]
    result=document_regions(image,detections)
    assert result[0].bbox[0]==0
    assert document_regions(image,[]) == []
    assert document_regions(image,[field('face',(1,1,10,10),'face')]) == []


def test_oversized_cluster_no_full_frame_region():
    image=np.zeros((120,180,3),np.uint8)
    detections=[field('id',(1,10,170,30)),field('name',(1,40,170,60)),field('dob',(1,75,170,110))]
    result=document_regions(image,detections)
    assert all(d.class_name=='document_boundary_unresolved' for d in result)
    assert all(d.bbox != (0,0,180,120) for d in result)


def test_adjacent_cards_with_separate_ids_not_merged():
    result=document_regions(np.zeros((300,500,3),np.uint8),fields()+fields(140))
    assert len(result)==2
    assert all(d.class_name=='document_region' for d in result)
    assert all(d.metadata['unique_fields']==3 for d in result)
