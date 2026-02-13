import os
from dagster import IOManager, InputContext, OutputContext, io_manager
from datasets import Dataset, load_from_disk


class HFDatasetIOManager(IOManager):
    """Persists HuggingFace Dataset objects to disk using Arrow format."""

    def __init__(self, base_dir: str = "./dagster_storage"):
        self._base_dir = base_dir

    def _get_path(self, context) -> str:
        return os.path.join(self._base_dir, *context.asset_key.path)

    def handle_output(self, context: OutputContext, obj):
        if obj is None:
            return
        path = self._get_path(context)
        if isinstance(obj, Dataset):
            obj.save_to_disk(path)
            context.log.info(f"Saved dataset ({len(obj)} rows) to {path}")
        elif isinstance(obj, dict):
            for split_name, ds in obj.items():
                split_path = os.path.join(path, split_name)
                ds.save_to_disk(split_path)
                context.log.info(
                    f"Saved {split_name} split ({len(ds)} rows) to {split_path}"
                )
        else:
            raise TypeError(f"Unsupported type: {type(obj)}")

    def load_input(self, context: InputContext):
        path = self._get_path(context)
        context.log.info(f"Loading dataset from {path}")
        return load_from_disk(path)


@io_manager
def hf_dataset_io_manager(_):
    return HFDatasetIOManager()
