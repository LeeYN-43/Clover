from .base import BaseHead
from .ssl_head import NCEHeadForMM, NCEHeadForText, NCEHeadForVision
from .mlm_itm_head import ITMHead, MLMHead
from .qa_head import QA_OE_Head, QA_MC_head

__all__ = [
    'BaseHead', 'NCEHeadForMM', 'NCEHeadForText', 'NCEHeadForVision',
    'MLMHead', 'WTIHeadForRetrieval', 'QA_OE_Head', 'QA_MC_head',
    'ITMHead',
]
