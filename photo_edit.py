import os
from PIL import Image

def process_and_add_images(source_folder, target_folder, size=(48, 48)):
    """
    Processes images in the source folder, resizes them, converts them to greyscale, and copies them to the target folder.
    Then renames all images in the target folder sequentially.

    :param source_folder: The path of the folder containing the new images.
    :param target_folder: The path of the target folder where the processed images should be copied.
    :param size: A tuple specifying the size (width, height) in pixels.
    """
    try:
        # Ensure the target folder exists
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # Process and copy images
        for filename in os.listdir(source_folder):
            file_path = os.path.join(source_folder, filename)
            
            # Check if the file is an image (you can add more extensions if needed)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                with Image.open(file_path) as img:
                    # Resize the image
                    img = img.resize(size, Image.Resampling.LANCZOS)
                    # Convert to greyscale
                    img = img.convert('L')
                    # Create the target path
                    target_path = os.path.join(target_folder, filename)
                    # Save the processed image to the target folder
                    img.save(target_path)
                    print(f"Image {filename} has been processed and copied to {target_folder}")
        
        # Rename all images in the target folder
        rename_images_in_folder(target_folder)
    except Exception as e:
        print(f"An error occurred: {e}")

def rename_images_in_folder(folder_path):
    """
    Renames all images in the folder sequentially as foldername_1, foldername_2, etc.

    :param folder_path: The path of the folder containing the images to be renamed.
    """
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
