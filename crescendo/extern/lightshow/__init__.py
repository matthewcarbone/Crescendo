try:
    from ._lightshow import prepare_dataset, save_dataset  # noqa
except ImportError as e:
    print(
        f"crescendo.extern.lightshow import failed with errors: {e}. Did you "
        'run\n pip install crescendo ".[lightshow]"\n or install required '
        "dependencies?"
    )
