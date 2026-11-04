import importlib.util
from pathlib import Path
import numpy as np
from privacy_vision.schemas import DocumentRegion

spec=importlib.util.spec_from_file_location('evaluation',Path(__file__).resolve().parents[2]/'scripts/evaluate_documents.py')
evaluation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_matching_one_to_one_and_class_aware():
    region=DocumentRegion((0,0,10,10),.8,'identity_card','detector')
    truth=[{'bbox':[0,0,10,10],'document_type':'identity_card'}]
    assert evaluation.matched_counts([region,region],truth)=={'tp':1,'fn':0,'fp':1}
    truth[0]['document_type']='birth_certificate'
    assert evaluation.matched_counts([region],truth)=={'tp':0,'fn':1,'fp':1}


def test_coverage_union_avoids_double_counting():
    target=evaluation.mask([(0,0,10,10),(5,0,15,10)],(20,20,3))
    covered=evaluation.mask([(0,0,5,10)],(20,20,3))
    assert target.sum()==150
    assert (target & covered).sum()/target.sum()==1/3
