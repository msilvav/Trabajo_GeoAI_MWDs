import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import shapes
from shapely.geometry import shape, Polygon, Point
import geopandas as gpd
from pycocotools.coco import COCO
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from openpyxl import Workbook
from openpyxl.worksheet.datavalidation import DataValidation

# Define la ruta de las imágenes y el archivo de anotaciones
image_directory = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/images'
annotation_file = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/result.json'
output_file = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/labeledMasks_individual.gpkg'
excel_output_file = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/masks_review.xlsx'

# Asegúrate de que la carpeta de salida exista
if os.path.exists(output_file):
    os.remove(output_file)  # Eliminar si ya existe para evitar conflictos

# Instancia COCO
coco = COCO(annotation_file)

# Colores distintivos para cada clase (puedes personalizarlos)
class_colors = {
    "class1": "#FF0000",  # Rojo
    "class2": "#00FF00",  # Verde
    "class3": "#0000FF",  # Azul
    # Añadir más colores según el número de clases
}

# Función para extraer coordenadas y fechas del nombre del archivo
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

# Función para procesar y devolver los resultados sin escribir
def process_image(image_id):
    img_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(image_directory, img_info['file_name'])
    img = rasterio.open(img_path).read(1)  # Lee la imagen como una sola banda (grayscale)
    
    coords, date_range, year = extract_coordinates_and_dates(img_info['file_name'])
    
    # Filtrar solo las imágenes del año 2018 (puedes ajustar el filtro según sea necesario)
    if year != "2018":
        return []
    
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)

    results = []
    for ann in anns:
        # Extraer la clase de la anotación
        class_id = ann['category_id']
        class_name = coco.loadCats(class_id)[0]['name']
        color = class_colors.get(class_name, "#FFFFFF")  # Color por defecto blanco
        
        # Crear la máscara correspondiente a la anotación
        mask = coco.annToMask(ann)
        
        # Generar la transformación a partir de las coordenadas de la imagen original
        transform = extract_coordinates_and_transform(coords, mask.shape[1], mask.shape[0])
        
        # Extraer los polígonos de la máscara
        for shape_data, value in shapes(mask, transform=transform):
            if value == 1:  # Considerar solo las áreas con valor 1 en la máscara
                polygon = shape(shape_data)
                centerpoint = polygon.centroid
                
                # Crear el ID único combinado
                unique_id = f"uniqueID{image_id}_annotationID{ann['id']}"
                
                # Crear enlace a Google Maps
                google_maps_link = f"https://www.google.com/maps/search/?api=1&query={centerpoint.y},{centerpoint.x}"
                
                # Definir los atributos para esta máscara
                attributes = {
                    "id": unique_id,
                    "class": class_name,
                    "filename": img_info['file_name'],
                    "year": year,
                    "color": color,
                    "centerpoint": centerpoint,
                    "google_maps_link": google_maps_link,
                    "review": "Pendiente"  # Valor inicial para el combobox
                }
                
                results.append((class_name, polygon, attributes))
    
    return results

# Función para agregar las máscaras a GeoPackage secuencialmente y exportar a Excel
def add_masks_to_geopackage(results, excel_data):
    for result in results:
        for class_name, polygon, attributes in result:
            # Convertir el polígono en un GeoDataFrame
            gdf = gpd.GeoDataFrame([attributes], geometry=[polygon], crs="EPSG:4326")
            
            # Nombre único para cada subcapa basada en la clase y el ID
            sublayer_name = f"{class_name}_{attributes['id']}"
            
            # Escribir en el GeoPackage de forma secuencial
            gdf.to_file(output_file, layer=sublayer_name, driver="GPKG", mode='a')
            
            # Agregar los datos relevantes al Excel
            excel_data.append([
                attributes['id'],
                attributes['filename'],
                attributes['class'],
                attributes['google_maps_link'],
                attributes['review']
            ])

# Procesamiento paralelo de imágenes
def process_images_parallel(image_ids):
    all_results = []
    excel_data = [["ID", "Filename", "Tipo", "Centerpoint", "Review"]]  # Encabezados del Excel

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, image_id) for image_id in image_ids]
        for future in as_completed(futures):
            all_results.append(future.result())  # Agregar resultados

    # Escribir resultados de forma secuencial y preparar datos para Excel
    add_masks_to_geopackage(all_results, excel_data)
    
    # Guardar Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "Revisión de Máscaras"

    for row in excel_data:
        ws.append(row)

    # Crear el combobox en la columna Review usando DataValidation
    dv = DataValidation(
        type="list",
        formula1='"Correcta,Modificar,Borrar,Pendiente"',
        showDropDown=True
    )
    
    # Aplicar la validación de datos a la columna E (Review)
    dv.add(f"E2:E{len(excel_data)}")
    
    # Añadir la validación de datos a la hoja
    ws.add_data_validation(dv)

    # Guardar el archivo Excel
    wb.save(excel_output_file)

# Obtener IDs de las imágenes
image_ids = coco.getImgIds()

# Procesar las imágenes en paralelo y escribir secuencialmente
process_images_parallel(image_ids)

print(f"GeoPackage generado en {output_file}")
print(f"Excel generado en {excel_output_file}")
