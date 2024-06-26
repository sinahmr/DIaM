import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco/coco-2014",
    splits=["validation", "test"],
    label_types=["segmentations"],
    max_samples=1000,
    shuffle=True,
    format=".jpg"
)
session = fo.launch_app(dataset)
session.dataset = dataset