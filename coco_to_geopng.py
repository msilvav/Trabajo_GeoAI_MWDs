import os
import re
import rasterio
from rasterio.mask import mask
import numpy as np
from rasterio.transform import from_bounds, Affine
from rasterio.errors import NotGeoreferencedWarning
from pycocotools.coco import COCO
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from shapely.geometry import shape, Polygon, MultiPolygon
from rasterio.windows import Window
import time
import shutil
import uuid

# Ignorar específicamente las advertencias de imágenes no georreferenciadas
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# Define las rutas necesarias
image_directory = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/images'
annotation_file = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/result.json'
output_directory = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/separado'
wld_directory = '/media/noobird/2002f002-8812-46a4-953d-1872302534b1/project-2-at-2024-06-17-17-54-839d4e00/wlds'

# Limpiar el directorio de salida antes de comenzar el procesamiento
for folder in [output_directory, wld_directory]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
# Asegúrate de que las carpetas de salida existan y estén limpias
os.makedirs(wld_directory, exist_ok=True)
os.makedirs(output_directory, exist_ok=True)

# Instancia COCO
coco = COCO(annotation_file)

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

# Convierte segmentación COCO a objeto Shapely
def convert_coco_poly_to_shapely(ann_segmentation):
    if isinstance(ann_segmentation, list):
        polygons = [Polygon(np.array(poly).reshape((-1, 2))) for poly in ann_segmentation if len(poly) >= 6]
        if len(polygons) == 1:
            return polygons[0]
        else:
            return MultiPolygon(polygons)
    else:
        raise ValueError("Formato de segmentación desconocido")

def reorganize_output():
    source_directory = output_directory
    destination_directory = os.path.join(source_directory, "../DS_Classifier")
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Recorrer cada subdirectorio y copiar los archivos PNG a la nueva estructura
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            if file.endswith(".png"):
                # Extraemos el nombre de la clase asegurándonos de ignorar otros subdirectorios
                parts = root.split(os.sep)
                class_name_index = parts.index('separado') + 1  # Asumiendo que 'separado' es el directorio justo antes del nombre de la clase
                class_name = parts[class_name_index]
                
                class_folder = os.path.join(destination_directory, class_name)
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)
                shutil.copy2(os.path.join(root, file), class_folder)

    print(f"Archivos reorganizados en {destination_directory}")

# Procesa cada imagen y sus anotaciones
def process_image(image_id):
    img_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(image_directory, img_info['file_name'])
    with rasterio.open(img_path) as src:
        coords, date_range, year = extract_coordinates_and_dates(img_info['file_name'])
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            class_id = ann['category_id']
            class_name = coco.loadCats(class_id)[0]['name']

            # Crea un identificador único para cada anotación procesada
            unique_dir = uuid.uuid4().hex
            class_dir = os.path.join(output_directory, class_name, year, unique_dir)
            os.makedirs(class_dir, exist_ok=True)

            polygon = convert_coco_poly_to_shapely(ann['segmentation'])
            masked_image, transformed_affine = mask(src, [polygon], crop=True)

            if masked_image.any():
                rows, cols = masked_image.shape[1:]
                window = rasterio.windows.Window(0, 0, cols, rows)
                transform = rasterio.windows.transform(window, transformed_affine)
                
                left, bottom, right, top = (
                    transform.c, transform.f + rows * transform.e, transform.c + cols * transform.a, transform.f)
                bbox_str = f"{top}, {left}, {bottom}, {right}"
                output_filename = f"{class_name}_({bbox_str}).png"
                output_path = os.path.join(class_dir, output_filename)

                try:
                    with rasterio.open(
                        output_path,
                        'w',
                        driver='PNG',
                        height=rows,
                        width=cols,
                        count=src.count,
                        dtype=masked_image.dtype,
                        crs=src.crs,
                        transform=transform
                    ) as dst:
                        for i in range(src.count):
                            dst.write(masked_image[i], i+1)
                except Exception as e:
                    print(f"Error al guardar la imagen {output_filename}: {e}")

            return f"Procesado: {img_info['file_name']}"

    return None

# Paralelizar el procesamiento de imágenes
def main():
    start_time = time.time()
    image_ids = coco.getImgIds()
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, image_id) for image_id in image_ids]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # Reorganizar y copiar los archivos al completar todos los procesos
    reorganize_output()

    print("\n".join([res for res in results if res]))
    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print(f"Tiempo de ejecución del script: {execution_time:.2f} minutos")

if __name__ == "__main__":
    main()

