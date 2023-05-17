import torchvision

image_size = 112
aihub_mean = [0.5444, 0.4335, 0.3800]
aihub_std = [0.2672, 0.2295, 0.2156]

nia_train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomApply(
            [
                torchvision.transforms.RandomAffine(degrees=10, shear=16),
                torchvision.transforms.RandomHorizontalFlip(p=1.0),
            ],
            p=0.5,
        ),
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop((224, 224)),
        torchvision.transforms.ToTensor(),
    ]
)

nia_valid_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ]
)

aihub_train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(image_size, image_size)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=aihub_mean, std=aihub_std),
    ]
)

aihub_valid_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(image_size, image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=aihub_mean, std=aihub_std),
    ]
)

aihub_test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(size=(image_size, image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=aihub_mean, std=aihub_std),
    ]
)