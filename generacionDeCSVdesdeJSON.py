import os
import re
import csv
import cv2
import numpy as np
from pyproj import Proj, Transformer
from pycocotools.coco import COCO
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define la ruta de las imágenes y el archivo de anotaciones
image_directory = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/images'
annotation_file = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/result.json'
output_csv = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/export_annotations.csv'


# Configuración de la proyección UTM y lat/long para Chile
zone = 19  # Huso horario
is_southern_hemisphere = True  # Hemisferio sur

utm_proj = Proj(proj='utm', zone=zone, south=is_southern_hemisphere, ellps='WGS84')
latlon_proj = Proj(proj='latlong', ellps='WGS84')
transformer_to_latlon = Transformer.from_proj(utm_proj, latlon_proj)

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

# Cálculo del paso en grados por píxel
def calculate_step(coords, img_width, img_height):
    lat_min, lon_min, lat_max, lon_max = map(float, coords.split(', '))
    lat_step = (lat_max - lat_min) / img_height
    lon_step = (lon_max - lon_min) / img_width
    return lat_step, lon_step, lat_min, lon_min

# Cálculo de coordenadas de bounding box en grados
def calculate_bbox_coordinates(lat_step, lon_step, lat_min, lon_min, bbox):
    x, y, w, h = bbox
    new_lat_min = lat_min + y * lat_step
    new_lon_min = lon_min + x * lon_step
    new_lat_max = new_lat_min + h * lat_step
    new_lon_max = new_lon_min + w * lon_step
    return new_lat_min, new_lon_min, new_lat_max, new_lon_max

# Cálculo del punto central del bounding box en lat/lon
def calculate_center_point(lat_min, lon_min, lat_max, lon_max):
    # Promediar las coordenadas mínimas y máximas para obtener el punto central
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    
    # No necesitamos convertir a UTM porque queremos las coordenadas en lat/lon para Google Maps
    return center_lat, center_lon

# Generación del nuevo nombre de archivo
def generate_new_filename(class_name, new_coords, date_range):
    return f"{class_name}_{new_coords} - ('{date_range[0]}', '{date_range[1]}').png"

# Lista para almacenar las filas del CSV
csv_rows = []

# Procesamiento de imágenes
def process_image(image_id):
    img_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(image_directory, img_info['file_name'])
    img = cv2.imread(img_path)
    
    coords, date_range, year = extract_coordinates_and_dates(img_info['file_name'])
    if not coords or not date_range or not year:
        print(f"Error al extraer datos del nombre del archivo: {img_info['file_name']}")
        return
    
    lat_step, lon_step, lat_min, lon_min = calculate_step(coords, img.shape[1], img.shape[0])
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        bbox = ann['bbox']
        class_id = ann['category_id']
        class_name = coco.loadCats(class_id)[0]['name']
        
        new_lat_min, new_lon_min, new_lat_max, new_lon_max = calculate_bbox_coordinates(lat_step, lon_step, lat_min, lon_min, bbox)
        
        # Calcular el punto central en lat/lon
        center_lat, center_lon = calculate_center_point(new_lat_min, new_lon_min, new_lat_max, new_lon_max)
        
        new_coords = f"[{new_lat_min:.5f}, {new_lon_min:.5f}, {new_lat_max:.5f}, {new_lon_max:.5f}]"
        new_filename = generate_new_filename(class_name, new_coords, date_range)
        
        csv_rows.append([
            date_range[0], date_range[1], year, class_name,
            coords, new_coords, f"({center_lat:.5f}, {center_lon:.5f})", new_filename
        ])

# Verificar si el archivo CSV ya existe
if not os.path.exists(output_csv):
    # Obtener IDs de las imágenes
    image_ids = coco.getImgIds()

    # Usar ThreadPoolExecutor para procesar imágenes en paralelo
    num_threads = 12
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_image, image_id): image_id for image_id in image_ids}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error procesando la imagen ID {futures[future]}: {e}")

    # Escribir el CSV
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Start Date', 'End Date', 'Year', 'Class', 'Original Bbox', 'Detected Bbox', 'Center Point (Lat/Lon)', 'New Filename'])
        csv_writer.writerows(csv_rows)

    print("Procesamiento completado y CSV generado.")
else:
    print(f"El archivo CSV ya existe en la ruta {output_csv}. No se generó un nuevo archivo.")
