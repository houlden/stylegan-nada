import dlib
from PIL import Image
from pathlib import Path

from src.utils.weights import download_shape_predictor_weights

from third_party.e4e.alignment import align_face


class FFHQFaceAligner:
    def __init__(self, weights_dir: Path | str = 'weights') -> None:
        predictor_weights = download_shape_predictor_weights(weights_dir=weights_dir)
        self.predictor = dlib.shape_predictor(str(predictor_weights))
    
    def __call__(self, image_path: Path | str) -> Image.Image:
        image_path = Path(image_path)
        
        assert image_path.exists(), f'Image not found: {image_path}'
        
        aligned_image = align_face(filepath=str(image_path), predictor=self.predictor)
        
        return aligned_image