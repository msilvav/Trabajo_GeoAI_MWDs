import os
import numpy as np
import rasterio
import re
from rasterio.transform import from_bounds
from pycocotools.coco import COCO

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
    # Corregir el orden de las coordenadas: lat_min, lon_min, lat_max, lon_max
    lon_min, lat_min, lon_max, lat_max = map(float, coords.split(', '))
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
    return transform

# Procesamiento de imágenes
def process_image(image_id):
    img_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(image_directory, img_info['file_name'])
    img = rasterio.open(img_path).read(1)  # Lee la imagen como una sola banda (grayscale)
    
    coords, date_range, year = extract_coordinates_and_dates(img_info['file_name'])
    
    # Filtrar solo las imágenes del año 2018
    if year != "2018":
        return
    
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    # Crear una máscara vacía con las mismas dimensiones que la imagen
    mask = np.zeros(img.shape, dtype=np.uint8)

    for ann in anns:
        # Dibujar la máscara correspondiente a la anotación
        mask += coco.annToMask(ann)

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
        'crs': 'EPSG:4326',  # CRS para WGS84
        'transform': transform
    }
    
    # Guardar la máscara como un GeoTIFF georreferenciado
    output_path = os.path.join(output_directory, f"{os.path.splitext(img_info['file_name'])[0]}_mask_2018.tif")
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(mask, 1)

    print(f"Máscara exportada a {output_path}")

# Obtener IDs de las imágenes
image_ids = coco.getImgIds()

# Procesar solo las imágenes del año 2018
for image_id in image_ids:
    process_image(image_id)
