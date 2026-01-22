import albumentations as A


class AlbBridge(A.ImageOnlyTransform):
    """
    Bridges an Albumentations transform (Compose or Sequential) to be used inside A.Compose.
    """

    def __init__(self, alb_transform, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.alb_transform = alb_transform

    def apply(self, image, **params):
        return self.alb_transform(image=image)["image"]

    def get_transform_init_args_names(self):
        return ("alb_transform",)


live_aug_pipeline = AlbBridge(
    A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.1,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.1,
            ),
            A.ImageCompression(
                quality_lower=40,
                quality_upper=60,
                p=0.3,
            ),
        ]
    )
)
