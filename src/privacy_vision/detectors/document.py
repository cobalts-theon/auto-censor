"""Bộ phát hiện toàn bộ phôi tài liệu và thẻ (Whole-document YOLO Adapter).

Hỗ trợ các loại tài liệu/thẻ: căn cước công dân, bằng lái xe, thẻ ngân hàng, giấy tờ.
"""
from .yolo import YoloDetector

DOCUMENT_CLASSES = {
    'identity_card', 'health_insurance_card', 'driver_license',
    'birth_certificate', 'document_page',
    'card_document', 'paper_document',
}


class DocumentDetector(YoloDetector):
    """Mô hình phát hiện khung bao tài liệu nguyên vẹn trên ảnh."""
    def __init__(self, config, device):
        super().__init__(config, device)
        names = self.model.names
        names = names.values() if isinstance(names, dict) else names
        if not names or not set(names) <= DOCUMENT_CLASSES:
            raise ValueError('Kiểu phân loại taxonomy của mô hình tài liệu không được hỗ trợ')
