import os
import shutil

from sklearn.model_selection import train_test_split

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def reorganize_llvip_dataset(
    src_root='LLVIP',
    dst_root='llvip_mod',
    train_val_split=0.2,
    random_seed=42
):
    # Ścieżki źródłowe
    src_visible_train = os.path.join(src_root, 'visible', 'train')
    src_infrared_train = os.path.join(src_root, 'infrared', 'train')
    src_visible_test = os.path.join(src_root, 'visible', 'test')
    src_infrared_test = os.path.join(src_root, 'infrared', 'test')
    src_annotations = os.path.join(src_root, 'Annotations')

    # Ścieżki docelowe
    dst_train_rgb = os.path.join(dst_root, 'train', 'rgb')
    dst_train_ir = os.path.join(dst_root, 'train', 'ir')
    dst_val_rgb = os.path.join(dst_root, 'val', 'rgb')
    dst_val_ir = os.path.join(dst_root, 'val', 'ir')
    dst_test_rgb = os.path.join(dst_root, 'test', 'rgb')
    dst_test_ir = os.path.join(dst_root, 'test', 'ir')
    dst_annotations = os.path.join(dst_root, 'Annotations')

    # Tworzenie folderów docelowych
    for path in [dst_train_rgb, dst_train_ir, dst_val_rgb, dst_val_ir, dst_test_rgb, dst_test_ir, dst_annotations]:
        create_dir_if_not_exists(path)

    # Pobierz listę obrazów treningowych
    train_images = sorted(os.listdir(src_visible_train))
    train_images = [img for img in train_images if img.endswith('.jpg')]

    # Podział na train i val
    train_images, val_images = train_test_split(
        train_images,
        test_size=train_val_split,
        random_state=random_seed
    )

    # Kopiowanie obrazów treningowych
    for img in train_images:
        # RGB
        src_rgb = os.path.join(src_visible_train, img)
        dst_rgb = os.path.join(dst_train_rgb, img)
        shutil.copy2(src_rgb, dst_rgb)
        # IR
        src_ir = os.path.join(src_infrared_train, img)
        dst_ir = os.path.join(dst_train_ir, img)
        shutil.copy2(src_ir, dst_ir)
        # Annotacje
        annotation = img.replace('.jpg', '.txt')
        src_ann = os.path.join(src_annotations, annotation)
        dst_ann = os.path.join(dst_annotations, annotation)
        if os.path.exists(src_ann):
            shutil.copy2(src_ann, dst_ann)

    # Kopiowanie obrazów walidacyjnych
    for img in val_images:
        # RGB
        src_rgb = os.path.join(src_visible_train, img)
        dst_rgb = os.path.join(dst_val_rgb, img)
        shutil.copy2(src_rgb, dst_rgb)
        # IR
        src_ir = os.path.join(src_infrared_train, img)
        dst_ir = os.path.join(dst_val_ir, img)
        shutil.copy2(src_ir, dst_ir)
        # Annotacje
        annotation = img.replace('.jpg', '.txt')
        src_ann = os.path.join(src_annotations, annotation)
        dst_ann = os.path.join(dst_annotations, annotation)
        if os.path.exists(src_ann):
            shutil.copy2(src_ann, dst_ann)

    # Kopiowanie obrazów testowych
    test_images = sorted(os.listdir(src_visible_test))
    test_images = [img for img in test_images if img.endswith('.jpg')]
    for img in test_images:
        # RGB
        src_rgb = os.path.join(src_visible_test, img)
        dst_rgb = os.path.join(dst_test_rgb, img)
        shutil.copy2(src_rgb, dst_rgb)
        # IR
        src_ir = os.path.join(src_infrared_test, img)
        dst_ir = os.path.join(dst_test_ir, img)
        shutil.copy2(src_ir, dst_ir)
        # Annotacje
        annotation = img.replace('.jpg', '.txt')
        src_ann = os.path.join(src_annotations, annotation)
        dst_ann = os.path.join(dst_annotations, annotation)
        if os.path.exists(src_ann):
            shutil.copy2(src_ann, dst_ann)

    print(f"Reorganizacja zakończona. Nowa struktura zapisana w: {dst_root}")
    print(f"Train: {len(train_images)} par")
    print(f"Val: {len(val_images)} par")
    print(f"Test: {len(test_images)} par")

if __name__ == "__main__":
    reorganize_llvip_dataset()