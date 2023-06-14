from typing import List, Dict

LARGE_OBJECT_CLASSES = ["laptop", "lamp", "camera"]
SMALL_OBJECT_CLASSES = [
    "bottle",
    "bowl",
    "can",
    "mug",
    "computer keyboard",
    "cellular telephone",
]

NOCS_CATEGORIES = ["bottle", "bowl", "can", "camera", "laptop", "mug"]
# NOCS_CATEGORIES = ["laptop"]
# NOCS_CATEGORIES = ["camera"]

## TODO Nick: Need to figure out where does this come from
SNC_SYNTH_ID_TO_CATEGORY: Dict[str, str] = {
    "02691156": "airplane",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02747177": "can",
    "02942699": "camera",
    "02954340": "cap",
    "02958343": "car",
    "03001627": "chair",
    "03046257": "clock",
    "03207941": "dishwasher",
    "03211117": "display",
    "04379243": "table",
    "04401088": "telephone",
    "02946921": "can",
    "04460130": "tower",
    "04468005": "train",
    "03085013": "computer keyboard",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "loudspeaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote control",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04530566": "vessel",
    "04554684": "washer",
    "02992529": "cellular telephone",
}


## TODO Nick: This could potentially be refactored
def shapenet_id_to_shape(categories: List[str]):
    id_to_shapes = {}
    ids = []

    for i, (shape_id, shape_name) in enumerate(SNC_SYNTH_ID_TO_CATEGORY.items()):
        if shape_name not in categories:
            continue
        print("category name", shape_name)
        ids.append(shape_id)
        id_to_shapes[i] = shape_name

    return id_to_shapes, ids
