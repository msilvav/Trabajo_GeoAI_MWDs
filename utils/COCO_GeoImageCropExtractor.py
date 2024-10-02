import os
import re
import rasterio
import numpy as np
from rasterio.transform import from_bounds
from rasterio.windows import Window
from PIL import Image
from pycocotools.coco import COCO
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from collections import Counter
import logging
from typing import Tuple, List, Dict, Optional

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_coordinates_and_dates(filename: str) -> Tuple[Optional[str], Optional[Tuple[str, str]], Optional[str]]:
    """
    Extrae coordenadas, rango de fechas y año de un nombre de archivo.
    
    Args:
        filename (str): Nombre del archivo a procesar.
    
    Returns:
        Tuple[Optional[str], Optional[Tuple[str, str]], Optional[str]]: 
            Coordenadas, rango de fechas y año extraídos, o None si no se encuentran.
    """
    pattern = r"\[([^\]]+)\] - \('([^']+)', '([^']+)'\) - (\w+)"
    match = re.search(pattern, filename)
    if match:
        coords = match.group(1)
        date_range = (match.group(2), match.group(3))
        year = match.group(2)[:4]
        return coords, date_range, year
    logger.warning(f"No se pudieron extraer coordenadas para {filename}")
    return None, None, None

def extract_coordinates_and_transform(coords: str, width: int, height: int) -> rasterio.transform.Affine:
    """
    Extrae coordenadas y crea una transformación geoespacial.
    
    Args:
        coords (str): String de coordenadas en formato "lon_min, lat_min, lon_max, lat_max".
        width (int): Ancho de la imagen.
        height (int): Altura de la imagen.
    
    Returns:
        rasterio.transform.Affine: Transformación geoespacial.
    """
    lon_min, lat_min, lon_max, lat_max = map(float, coords.split(', '))
    return from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)

def process_single_image(coco: COCO, img_info: Dict, image_directory: str, output_directory: str) -> Tuple[List[str], Counter]:
    """
    Procesa una sola imagen, extrayendo recortes de objetos anotados.
    
    Args:
        coco (COCO): Objeto COCO cargado.
        img_info (Dict): Información de la imagen a procesar.
        image_directory (str): Directorio de las imágenes originales.
        output_directory (str): Directorio para guardar los recortes.
    
    Returns:
        Tuple[List[str], Counter]: Lista de rutas de recortes guardados y contador de categorías.
    """
    img_id = img_info['id']
    img_filename = img_info['file_name']
    img_path = os.path.join(image_directory, img_filename)
    logger.info(f"Procesando imagen: {img_filename}")

    coords, date_range, year = extract_coordinates_and_dates(img_filename)
    if coords is None:
        return [], Counter()

    results = []
    category_counts = Counter()
    try:
        with rasterio.open(img_path) as src:
            transform = extract_coordinates_and_transform(coords, src.width, src.height)
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            for ann in anns:
                bbox = ann['bbox']
                category_id = ann['category_id']
                category_name = coco.loadCats([category_id])[0]['name']
                
                category_output_dir = os.path.join(output_directory, category_name)
                os.makedirs(category_output_dir, exist_ok=True)

                window = Window(bbox[0], bbox[1], bbox[2], bbox[3])
                crop = src.read(window=window)

                # Calcular las coordenadas geográficas del recorte
                crop_bounds = rasterio.windows.bounds(window, transform)
                crop_coords = f"{crop_bounds[0]:.5f},{crop_bounds[1]:.5f},{crop_bounds[2]:.5f},{crop_bounds[3]:.5f}"

                # Manejo de diferentes formatos de imagen
                if crop.shape[0] == 1:
                    pil_image = Image.fromarray(crop[0], mode='L')
                elif crop.shape[0] == 3:
                    pil_image = Image.fromarray(np.transpose(crop, (1, 2, 0)), mode='RGB')
                else:
                    raise ValueError(f"Número de canales no soportado: {crop.shape[0]}")

                # Modificar el nombre del archivo para incluir las nuevas coordenadas
                crop_filename = f"[{crop_coords}]_{category_name}_{ann['id']}.png"
                crop_path = os.path.join(category_output_dir, crop_filename)
                pil_image.save(crop_path)
                results.append(crop_path)
                category_counts[category_name] += 1

    except Exception as e:
        logger.error(f"Error al procesar {img_filename}: {str(e)}")

    logger.info(f"Procesamiento completado para {img_filename}")
    return results, category_counts

def extract_image_crops_parallel(coco: COCO, image_directory: str, output_directory: str, max_workers: Optional[int] = None) -> Tuple[List[str], Counter]:
    """
    Procesa imágenes en paralelo, extrayendo recortes de objetos.
    
    Args:
        coco (COCO): Objeto COCO cargado.
        image_directory (str): Directorio de las imágenes originales.
        output_directory (str): Directorio para guardar los recortes.
        max_workers (Optional[int]): Número máximo de workers para el procesamiento paralelo.
    
    Returns:
        Tuple[List[str], Counter]: Lista de todas las rutas de recortes y contador total de categorías.
    """
    os.makedirs(output_directory, exist_ok=True)
    image_ids = coco.getImgIds()
    all_results = []
    total_category_counts = Counter()
    logger.info(f"Procesando {len(image_ids)} imágenes en paralelo...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, coco, coco.loadImgs([img_id])[0], image_directory, output_directory) for img_id in image_ids]
        for future in as_completed(futures):
            try:
                results, counts = future.result()
                all_results.extend(results)
                total_category_counts.update(counts)
            except Exception as e:
                logger.error(f"Error en el procesamiento paralelo: {str(e)}")
    
    logger.info(f"Procesamiento paralelo completado. Total de recortes: {len(all_results)}")
    return all_results, total_category_counts

# Uso del script
if __name__ == "__main__":
    start_time = time.time()

    # Configuración
    coco_json_path = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/result.json'
    image_directory = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/images'
    output_directory = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/DB_Separado'

    # Cargar el dataset COCO
    coco = COCO(coco_json_path)

    # Procesar imágenes y extraer recortes en paralelo
    results, category_counts = extract_image_crops_parallel(coco, image_directory, output_directory, max_workers=20)

    # Calcular el tiempo de ejecución
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60

    # Generar y mostrar el resumen
    logger.info("\n--- Resumen del Procesamiento ---")
    logger.info(f"Número total de imágenes procesadas: {len(coco.getImgIds())}")
    logger.info(f"Número total de recortes generados: {len(results)}")
    logger.info("\nRecortes generados por categoría:")
    for category, count in category_counts.items():
        logger.info(f"  - {category}: {count}")
    logger.info(f"\nTiempo total de ejecución: {execution_time_minutes:.2f} minutos")