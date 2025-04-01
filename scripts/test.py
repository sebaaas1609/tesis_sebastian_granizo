import os

# Directorios
carpeta_imagenes = 'data/images/train'
carpeta_labels = 'task_ladybug_boxes_annotations_2025_03_11_22_48_37_yolo 1.1/obj_train_data'


# Obtener nombres de archivos sin extensión
nombres_imagenes = {os.path.splitext(f)[0] for f in os.listdir(carpeta_imagenes)}
nombres_labels = {os.path.splitext(f)[0] for f in os.listdir(carpeta_labels)}

# Encontrar las imágenes que no tienen un label correspondiente
faltantes = nombres_imagenes - nombres_labels

if faltantes:
    print("Archivos en carpeta de imágenes pero no en carpeta de labels:")
    for nombre in faltantes:
        print(nombre)
else:
    print("No hay archivos faltantes.")