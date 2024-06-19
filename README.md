# CNN Model Training for Facial State Recognition

## Authors
- AK_14
- Rezeq Khader
- Mohammed Mousa
- Omar Hassanein

## Contents and Purpose of Each File

### class_distribution_plot.py
- **Purpose**: Generates bar plots to visualize the class distribution of images in the 'train' and 'test' datasets.
- **Key Functions**:
  - `plot_class_distribution(dataset_path, title)`: Plots the distribution of classes in a given dataset directory.
- **Dependencies**: torchvision.datasets, matplotlib.pyplot, seaborn.

### image_preprocessing.py
- **Purpose**: Processes and resizes images from a source folder, converts them to grayscale, and saves them to a target folder. It also renames the processed images.
- **Key Functions**:
  - `process_and_add_images(source_folder, target_folder, size=(48, 48))`: Processes and saves images.
  - `rename_images_in_folder(folder_path)`: Renames the images in the target folder.
- **Dependencies**: os, PIL.Image.

### pixel_intensity_distribution.py
- **Purpose**: Plots the pixel intensity distribution of images in the 'train' and 'test' datasets.
- **Key Functions**:
  - `plot_pixel_intensity_distribution(loader, title='Pixel Intensity Distribution')`: Plots the distribution of pixel intensities for each class.
- **Dependencies**: torch, torchvision.datasets, torch.utils.data.DataLoader, matplotlib.pyplot, numpy, collections.Counter.

### sample_pixel_intensity_histograms.py
- **Purpose**: Plots histograms of pixel intensities for a sample of images from each class in the 'train' and 'test' datasets.
- **Key Functions**:
  - `plot_sample_pixel_intensity_histograms(loader, num_samples_per_class=15)`: Plots images and their pixel intensity histograms for each class.

### cnn_variants.py
- **Purpose**: Defines and trains various Convolutional Neural Network (CNN) models for facial state recognition. It includes data preprocessing, model architecture definitions, and training routines.
- **Dependencies**: torch, torchvision.datasets, torch.utils.data.DataLoader, matplotlib.pyplot, numpy.

### predict.py
- **Purpose**: Loads a trained CNN model and uses it to make predictions on new images or datasets.
- **Dependencies**: torch, torchvision.datasets, torch.utils.data.DataLoader, matplotlib.pyplot, numpy.

## Dependencies

Ensure you have the following libraries installed:

- **numpy**
- **scikit-learn**
- **torch**
- **torchvision**
- **seaborn**
- **matplotlib**
- **collections-extended** (if additional collections are needed)
- **tabulate**

### Key Libraries and Modules
- **torch**
  - `torch.utils.data.DataLoader`
- **torchvision.datasets**
- **matplotlib.pyplot**
- **numpy**

## Steps to Execute the Code

### Data Cleaning
1. Ensure the required packages are installed:
    ```bash
    pip install torch torchvision pillow matplotlib seaborn numpy scikit-learn collections-extended tabulate
    ```
2. Run the `image_preprocessing.py` script:
    - **Description**: This script will resize the images in the source folder to 48x48 pixels, convert them to grayscale, save them to the target folder, and rename the images.
    - **Command**:
        ```bash
        python image_preprocessing.py
        ```
    - **Parameters**:
        - `source_folder`: The folder containing the original images.
        - `target_folder`: The folder where processed images will be saved.

### Data Visualization
1. Ensure the required packages are installed:
    ```bash
    pip install torch torchvision pillow matplotlib seaborn numpy scikit-learn collections-extended tabulate
    ```
2. Run the `class_distribution_plot.py` script:
    - **Description**: Generates bar plots to visualize the class distribution in the 'train' and 'test' datasets.
    - **Command**:
        ```bash
        python class_distribution_plot.py
        ```
3. Run the `pixel_intensity_distribution.py` script:
    - **Description**: Plots the pixel intensity distribution for each class in the 'train' and 'test' datasets.
    - **Command**:
        ```bash
        python pixel_intensity_distribution.py
        ```
4. Run the `sample_pixel_intensity_histograms.py` script:
    - **Description**: Plots sample images and their pixel intensity histograms for each class in the 'train' and 'test' datasets.
    - **Command**:
        ```bash
        python sample_pixel_intensity_histograms.py
        ```

### Model Training
1. Ensure the required packages are installed:
    ```bash
    pip install torch torchvision pillow matplotlib seaborn numpy scikit-learn collections-extended tabulate
    ```
2. Run the `cnn_variants.py` script to train the models:
    - **Description**: This script defines and trains the CNN models.
    - **Command**:
        ```bash
        python3 cnn_variants.py
        ```

By following these steps, you can preprocess the data, visualize the class distribution and pixel intensity distributions of your datasets, and train the CNN models.


