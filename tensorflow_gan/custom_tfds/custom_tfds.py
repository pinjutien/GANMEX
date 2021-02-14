import tensorflow_datasets as tfds
from pathlib import Path
import os


class obj_scene_v2(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for a2o dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    _DESCRIPTION = """"""
    _CITATION = """"""

    ROOT_DIR = "/home/ec2-user/bam/data/obj_scene_v2/"
    LABEL_NAMES = [
        "pizza-bamboo-forest",
        "pizza-bedroom",
        "stop-sign-bamboo-forest",
        "stop-sign-bedroom"
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=self._DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(None, None, 3)),
                'label': tfds.features.ClassLabel(names=self.LABEL_NAMES),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage='https://dataset-homepage/',
            citation=self._CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        train_path = Path(os.path.join(self.ROOT_DIR, "train"))
        test_path = Path(os.path.join(self.ROOT_DIR, "test"))

        return[
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={'path': train_path},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={'path': test_path},
            ),
        ]

    def _generate_examples(self, path):
        """Yields examples."""
        for f in path.glob('*.jpg'):
            yield f.name, {
                'image': str(f),
                'label':  f.name.split('_')[0]
            }
