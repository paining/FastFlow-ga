import os
import shutil
import random

import log
import logging

types = ["BEAD", "CRACK"]

good_path = ".data/skon/good"
defect_path = ".data/skon/fakedefect"

result_path = ".data/skon/dataset"

log.set_logger("log.yaml", os.path.join(result_path, "log.log"))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.makedirs(result_path, exist_ok=True)
for typename in types:
    subdirs = list(
        set(os.listdir(os.path.join(good_path, typename))) | 
        set(os.listdir(os.path.join(defect_path, typename)))
        )
    os.makedirs(os.path.join(result_path, typename), exist_ok=True)

    for subdir in subdirs:
        category = f"{typename}_{subdir}"
        os.makedirs(os.path.join(result_path, category), exist_ok=True)
        os.makedirs(os.path.join(result_path, category, "train"), exist_ok=True)
        os.makedirs(os.path.join(result_path, category, "train", "good"), exist_ok=True)
        os.makedirs(os.path.join(result_path, category, "test"), exist_ok=True)
        os.makedirs(os.path.join(result_path, category, "test", "good"), exist_ok=True)
        os.makedirs(os.path.join(result_path, category, "test", "defect"), exist_ok=True)
        goodfiles = os.listdir(os.path.join(good_path, typename, subdir))
        defectfiles = os.listdir(os.path.join(defect_path, typename, subdir))

        logger.info(f"{len(goodfiles):5d} good images and {len(defectfiles):5d} defect images in {typename}/{subdir}")
        if len(goodfiles) > 2*len(defectfiles):
            idxs = random.sample(range(len(goodfiles)), len(defectfiles))
        else:
            idxs = random.sample(range(len(goodfiles)), len(goodfiles)//2)

        logger.info(f"Good image splited by {len(goodfiles)-len(idxs):5d} train and {len(idxs):5d} test.")

        for i, file in enumerate(goodfiles):
            if i in idxs:
                shutil.copyfile(
                    os.path.join(good_path, typename, subdir, file), 
                    os.path.join(result_path, category, "test", "good", file)
                    )
            else:
                shutil.copyfile(
                    os.path.join(good_path, typename, subdir, file), 
                    os.path.join(result_path, category, "train", "good", file)
                    )
        for file in defectfiles:
            shutil.copyfile(
                os.path.join(defect_path, typename, subdir, file), 
                os.path.join(result_path, category, "test", "defect", file)
                )