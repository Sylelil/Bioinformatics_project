from pathlib import Path
from openslide import open_slide
from src.images.preprocessing import apply_filters_to_image

sample_img = Path('..') / '..' / '..' / 'sample_images' / 'svs' / 'sample.svs'

def remove_bg():
    slide = open_slide((str(sample_img)))
    slide_info = {
        "slide_name": 0,
        "slide_width": 1024,
        "slide_height": 1024
    }
    apply_filters_to_image(slide_info, slide.get_thumbnail((1024, 1024)), None, display=True)

if __name__ == '__main__':
    remove_bg()