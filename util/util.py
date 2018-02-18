def get_images_at_path(images_path):
    import os

    included_extensions = ['jpg', 'jpeg', 'bmp', 'png', 'gif']
    return [fn for fn in os.listdir(images_path) if any(fn.endswith(ext) for ext in included_extensions)]