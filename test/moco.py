import torch
from timm import create_model

from src.moco import MOCO


def test_moco_init():
    model = create_model('resnet50', num_classes=128)
    moco = MOCO(model)

    assert moco is not None
    assert moco.encoder_q is not None
    assert moco.encoder_k is not None
    assert moco.queue is None

def test_moco_forward():
    model = create_model('resnet50', num_classes=128)
    moco = MOCO(model)

    x = torch.rand([2, 3, 224, 224])
    x2 = torch.rand([2, 3, 224, 224])

    moco(x)
    moco(x2)

    assert int(moco.queue_ptr[0]) == 4
