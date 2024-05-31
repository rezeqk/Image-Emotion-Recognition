AK_14

 Rezeq khader
 Mohammed Mousa
 Omar Hassanein

Contents and Purpose of Each File
class_distribution_plot.py:

Purpose: This script generates bar plots to visualize the class distribution of images in the 'train' and 'test' datasets.
Key Functions:
plot_class_distribution(dataset_path, title): Plots the distribution of classes in a given dataset directory.
Dependencies: torchvision.datasets, matplotlib.pyplot, seaborn.
image_preprocessing.py:

Purpose: This script processes and resizes images from a source folder, converts them to grayscale, and saves them to a target folder. It also renames the processed images.
Key Functions:
process_and_add_images(source_folder, target_folder, size=(48, 48)): Processes and saves images.
rename_images_in_folder(folder_path): Renames the images in the target folder.
Dependencies: os, PIL.Image.
pixel_intensity_distribution.py:

Purpose: This script plots the pixel intensity distribution of images in the 'train' and 'test' datasets.
Key Functions:
plot_pixel_intensity_distribution(loader, title='Pixel Intensity Distribution'): Plots the distribution of pixel intensities for each class.
Dependencies: torch, torchvision.datasets, torch.utils.data.DataLoader, matplotlib.pyplot, numpy, collections.Counter.
sample_pixel_intensity_histograms.py:

Purpose: This script plots histograms of pixel intensities for a sample of images from each class in the 'train' and 'test' datasets.
Key Functions:
plot_sample_pixel_intensity_histograms(loader, num_samples_per_class=15): Plots images and their pixel intensity histograms for each class.
Dependencies: torch, torchvision.datasets, torch.utils.data.DataLoader, matplotlib.pyplot, numpy.
Steps to Execute the Code
Data Cleaning
To preprocess and clean the images:

Ensure the required packages are installed:
pip install torch torchvision pillow matplotlib seaborn numpy

Run the image_preprocessing.py script:

Description: This script will resize the images in the source folder to 48x48 pixels, convert them to grayscale, save them to the target folder, and rename the images.
Command:
python image_preprocessing.py
Parameters:
source_folder: The folder containing the original images.
target_folder: The folder where processed images will be saved.
Data Visualization
To visualize the class distribution and pixel intensity distribution:

Ensure the required packages are installed:
pip install torch torchvision pillow matplotlib seaborn numpy
Run the class_distribution_plot.py script:

Description: This script will generate bar plots to visualize the class distribution in the 'train' and 'test' datasets.
Command:
python class_distribution_plot.py
Run the pixel_intensity_distribution.py script:

Description: This script will plot the pixel intensity distribution for each class in the 'train' and 'test' datasets.
Command:
python pixel_intensity_distribution.py
Run the sample_pixel_intensity_histograms.py script:

Description: This script will plot sample images and their pixel intensity histograms for each class in the 'train' and 'test' datasets.
Command:
python sample_pixel_intensity_histograms.py
By following these steps, you can preprocess the data and visualize the class distribution and pixel intensity distributions of your datasets.



