import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pycocotools.coco import COCO
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define la ruta de las imágenes y el archivo de anotaciones
image_directory = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/images'
annotation_file = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/result.json'
output_directory = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/export_geotiffs_new'

# Asegúrate de que la carpeta de salida exista
os.makedirs(output_directory, exist_ok=True)

# Instancia COCO
coco = COCO(annotation_file)

# Función para extraer coordenadas, fechas y año del nombre del archivo
def extract_coordinates_and_dates(filename):
    pattern = r"\[([^\]]+)\] - \('([^']+)', '([^']+)'\) - (\w+)"
    match = re.search(pattern, filename)
    if match:
        coords = match.group(1)
        date_range = match.group(2), match.group(3)
        year = match.group(2)[:4]  # Extrae el año de la fecha inicial
        return coords, date_range, year
    return None, None, None

# Función para extraer las coordenadas y generar la transformación
def extract_coordinates_and_transform(coords, width, height):
    lon_min, lat_min, lon_max, lat_max = map(float, coords.split(', '))
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    return transform

# Procesamiento de imágenes y anotaciones
def process_annotation(image_id, ann):
    try:
        img_info = coco.loadImgs(image_id)[0]
        img_path = os.path.join(image_directory, img_info['file_name'])
        img = rasterio.open(img_path).read(1)  # Lee la imagen como una sola banda (grayscale)
        
        coords, date_range, year = extract_coordinates_and_dates(img_info['file_name'])
        
        # Filtrar solo las imágenes del año 2018
        if year != "2018":
            return
        
        # Extraer la clase de la anotación
        class_id = ann['category_id']
        class_name = coco.loadCats(class_id)[0]['name']
        
        # Crear la máscara correspondiente a la anotación
        mask = coco.annToMask(ann)
        
        # Generar la transformación a partir de las coordenadas de la imagen original
        transform = extract_coordinates_and_transform(coords, mask.shape[1], mask.shape[0])
        
        # Definir el perfil para el archivo GeoTIFF
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': 0,
            'width': mask.shape[1],
            'height': mask.shape[0],
            'count': 1,
            'crs': 'EPSG:4326',
            'transform': transform
        }
        
        # Simplificar el nombre del archivo para evitar errores de GDAL
        output_filename = f"{img_info['id']}_{class_name}_mask_2018.tif"
        output_path = os.path.join(output_directory, output_filename)
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(mask, 1)

        logging.info(f"Máscara exportada a {output_path}")

    except Exception as e:
        logging.error(f"Error procesando la imagen ID {image_id}: {e}")

# Procesamiento paralelo de las imágenes y anotaciones
def process_images_parallel(image_ids):
    total_tasks = sum(len(coco.getAnnIds(imgIds=image_id)) for image_id in image_ids)
    completed_tasks = 0
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_annotation, image_id, ann) 
                   for image_id in image_ids 
                   for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id))]
        for future in as_completed(futures):
            completed_tasks += 1
            logging.info(f"Tarea completada: {completed_tasks}/{total_tasks}")
            try:
                future.result()  # Propaga excepciones si las hay
            except Exception as e:
                logging.error(f"Error en tarea completada: {e}")

# Obtener IDs de las imágenes
image_ids = coco.getImgIds()

# Procesar las imágenes en paralelo
process_images_parallel(image_ids)
