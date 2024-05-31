import os
from PIL import Image

def process_and_add_images(source_folder, target_folder, size=(48, 48)):


    try:

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        

        for filename in os.listdir(source_folder):
            file_path = os.path.join(source_folder, filename)
            

            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                with Image.open(file_path) as img:

                    img = img.resize(size, Image.Resampling.LANCZOS)

                    img = img.convert('L')

                    target_path = os.path.join(target_folder, filename)

                    img.save(target_path)
                    print(f"Image {filename} has been processed and copied to {target_folder}")
        

        rename_images_in_folder(target_folder)
    except Exception as e:
        print(f"An error occurred: {e}")

def rename_images_in_folder(folder_path):



    try:
        folder_name = os.path.basename(folder_path)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        for index, filename in enumerate(sorted(image_files), start=1):
            file_path = os.path.join(folder_path, filename)
            new_filename = f"{folder_name}_{index}{os.path.splitext(filename)[1]}"
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(file_path, new_file_path)
            print(f"Renamed {filename} to {new_filename}")
    except Exception as e:
        print(f"An error occurred while renaming files: {e}")

# Example usage
source_folder = r'new images'
target_folder = r'train/focused'
process_and_add_images(source_folder, target_folder)
