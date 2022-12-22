from torchvision.datasets import ImageFolder

from .dataset import register_dataset

@register_dataset
class ImageFolder_with_Idx(ImageFolder):
    def __getitem__(self, item):
        previous_item = super().__getitem__(item)

        return previous_item, item