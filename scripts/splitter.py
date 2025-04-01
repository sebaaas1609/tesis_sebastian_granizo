import os
import shutil
import random

# Definir las carpetas
images_dir = '/Users/sebasgranizo/Developer/tesis/capstone_project_sebas_granizo/data/images/train'  # Ruta a la carpeta con las imágenes
labels_dir = '/Users/sebasgranizo/Developer/tesis/capstone_project_sebas_granizo/data/labels/train'  # Ruta a la carpeta con los archivos .txt
output_dir = '/Users/sebasgranizo/Developer/tesis/capstone_project_sebas_granizo/data'  # Carpeta donde se crearán train, val, test

# Crear las carpetas para train, val, test
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

for dir in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dir, 'labels'), exist_ok=True)

# Obtener los nombres de los archivos de imágenes con diferentes extensiones
image_extensions = ('.jpg', '.jpeg', '.png')
images = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]

# Mezclar los archivos aleatoriamente
random.shuffle(images)

# Definir los tamaños de los splits (70% train, 15% val, 15% test)
train_split = int(len(images) * 0.7)
val_split = int(len(images) * 0.15)

# Dividir en entrenamiento, validación y prueba
train_images = images[:train_split]
val_images = images[train_split:train_split+val_split]
test_images = images[train_split+val_split:]

# Función para mover las imágenes y sus archivos .txt correspondientes
def move_files(image_list, src_images_dir, src_labels_dir, dst_images_dir, dst_labels_dir):
    for image in image_list:
        # Mover imagen
        shutil.move(os.path.join(src_images_dir, image), os.path.join(dst_images_dir, image))
        
        # Mover archivo .txt correspondiente
        label_file = os.path.splitext(image)[0] + '.txt'  # Mantiene el nombre, cambia la extensión a .txt
        label_path = os.path.join(src_labels_dir, label_file)
        
        if os.path.exists(label_path):  # Solo mover si existe el archivo .txt
            shutil.move(label_path, os.path.join(dst_labels_dir, label_file))

# Mover los archivos a sus respectivos directorios
move_files(train_images, images_dir, labels_dir, os.path.join(train_dir, 'images'), os.path.join(train_dir, 'labels'))
move_files(val_images, images_dir, labels_dir, os.path.join(val_dir, 'images'), os.path.join(val_dir, 'labels'))
move_files(test_images, images_dir, labels_dir, os.path.join(test_dir, 'images'), os.path.join(test_dir, 'labels'))

print(f'Split completado: {len(train_images)} en train, {len(val_images)} en val, {len(test_images)} en test.')
