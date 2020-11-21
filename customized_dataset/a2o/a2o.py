"""a2o dataset."""

import tensorflow_datasets as tfds
from pathlib import Path
import os
import json

with open('/Users/pin-jutien/tfds-download/test_dataset/a2o/data.json') as f:
  data_config = json.load(f)

root_dir = data_config.get("root_dir")
label_name = data_config.get("label_name")
# TODO(a2o): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(a2o): BibTeX citation
_CITATION = """
"""


class A2o(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for a2o dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(a2o): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
            'label': tfds.features.ClassLabel(names=label_name),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(a2o): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')

    # TODO(a2o): Returns the Dict[split names, Iterator[Key, Example]]
    # root_dir = data_config.get("root_dir")# "/Users/pin-jutien/tfds-download/test_dataset/apple2orange"
    train_path = Path(os.path.join(root_dir, "train"))
    test_path = Path(os.path.join(root_dir, "test"))
    return {
        # 'train': self._generate_examples(path / 'train_imgs'),
        'train': self._generate_examples(train_path),
        'test': self._generate_examples(test_path),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(a2o): Yields (key, example) tuples from the dataset
    for f in path.glob('*.jpg'):
      yield f.name, {
          'image': f,
          'label':  "A" if f.name.startswith("A_") else "B",
      }
