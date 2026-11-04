"""Optional whole-document adapter. No model is downloaded or trained here."""
from .yolo import YoloDetector

DOCUMENT_CLASSES = {'identity_card', 'health_insurance_card', 'driver_license',
                    'birth_certificate', 'document_page',
                    'card_document', 'paper_document'}

class DocumentDetector(YoloDetector):
    def __init__(self, config, device):
        super().__init__(config, device)
        names = self.model.names
        names = names.values() if isinstance(names, dict) else names
        if not names or not set(names) <= DOCUMENT_CLASSES:
            raise ValueError('Unsupported document checkpoint taxonomy')
