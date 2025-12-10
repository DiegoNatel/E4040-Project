# ------------------------------------------------------------------------------
# File: data_utils.py
# Author: Bolun Li (UNI: bl3147)
# Note: All code in this file is my own implementation for the course project,
#       except for externally cited sources.
# ------------------------------------------------------------------------------

"""
data_utils.py

- Load and preprocess image datasets for DCGAN training
- Normalize images to [-1, 1] range for tanh activation
- Support for training datasets (LSUN, ImageNet, Faces) and evaluation datasets (CIFAR-10, SVHN, MNIST)
- Batch size: 128 for all datasets
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from pathlib import Path
import hashlib


### Weight Initialization Configuration
## All weights from N(0, 0.02) as per DCGAN paper

WEIGHT_INIT_CONFIG = {
    'distribution': 'normal',
    'mean': 0.0,
    'stddev': 0.02  
}


### Normalization Functions
def normalize_to_tanh_range(image):
    # Normalize image from [0, 255] to [-1, 1] for tanh activation
    if isinstance(image, tf.Tensor):
        image = tf.cast(image, tf.float32)
        return (image / 127.5) - 1.0
    else:
        return (image.astype(np.float32) / 127.5) - 1.0


def denormalize_from_tanh_range(tensor):
    # Inverse operation for visualization: maps [-1, 1] back to [0, 255]
    if isinstance(tensor, tf.Tensor):
        denorm = (tensor + 1.0) * 127.5
        return tf.cast(tf.clip_by_value(denorm, 0, 255), tf.uint8)
    else:
        denorm = ((tensor + 1.0) * 127.5).astype(np.float32)
        return np.clip(denorm, 0, 255).astype(np.uint8)

### LSUN Bedrooms Loader
## Output: 64×64 per DCGAN paper
## Includes deduplication step as per section 4.1.1

def _compute_semantic_hash_numpy(image_np):
    """Compute Average Hash (aHash) of image for semantic deduplication.
    
    This serves as a lightweight alternative to the Autoencoder-based 
    semantic hashing described in the DCGAN paper (Section 4.1.1).
    It detects near-duplicates by comparing structural features on a 
    downsampled 32x32 grayscale version of the image.
    """
    # Ensure image is 32x32 (input size per paper spec)
    # If input is not 32x32, we rely on the caller to have resized it
    # or we resize here (but for efficiency we assume 32x32 input from caller)
    
    # 1. Convert to grayscale (simple average of channels)
    if image_np.shape[-1] == 3:
        gray = np.mean(image_np, axis=-1)
    else:
        gray = image_np.squeeze()
        
    # 2. Compute mean value
    avg = gray.mean()
    
    # 3. Threshold to create binary hash (1 if > avg, 0 otherwise)
    # This captures low-frequency structural information
    binary_map = (gray > avg).astype(np.uint8)
    
    # 4. Pack bits into hash string
    # Flatten and pack into bytes for efficient storage/comparison
    hash_bytes = np.packbits(binary_map.flatten()).tobytes()
    
    # Return as hex string
    return hashlib.md5(hash_bytes).hexdigest()


def _deduplicate_dataset(dataset, target_size):
    """Remove duplicate images from dataset based on semantic hash.
    
    Fixed to use a streaming generator to prevent OOM errors on large datasets (LSUN).
    Maintains a set of seen hashes in memory (~100MB for 3M images) but streams 
    the images themselves.
    """
    
    def generator():
        # Keep track of seen hashes (RAM usage: ~32 bytes * 3M approx 100MB)
        seen_hashes = set()
        
        # Iterate over the dataset one by one
        for image in dataset:
            # Convert to numpy for hashing
            image_np = image.numpy()
            
            # Create 32x32 version for semantic hashing (per paper spec)
            if image_np.shape[0] != 32 or image_np.shape[1] != 32:
                # Simple subsampling for efficiency (64->32 is 2x)
                if image_np.shape[0] == 64 and image_np.shape[1] == 64:
                     image_small = image_np[::2, ::2, :]
                else:
                     image_small = image_np
            else:
                image_small = image_np
                
            # Compute semantic hash
            hash_str = _compute_semantic_hash_numpy(image_small)
            
            # Yield only if new
            if hash_str not in seen_hashes:
                seen_hashes.add(hash_str)
                yield image_np

    # Create a new dataset from the generator
    # We use the element_spec from the input dataset to ensure types match
    return tf.data.Dataset.from_generator(
        generator,
        output_signature=dataset.element_spec
    )


def load_lsun(category='bedroom', batch_size=128, shuffle_buffer=None, 
              target_size=(64, 64), shuffle=True, deduplicate=True):

    # Pipeline: Load → Resize to 64×64 → Deduplicate → Normalize to [-1, 1] → Batch of 128
    assert batch_size == 128, "Batch size must be 128 per specification"
    assert target_size == (64, 64), "LSUN output must be 64x64 per specification"
    
    try:
        ds = tfds.load(f'lsun/{category}', split='train', as_supervised=False)
        
        def extract_image(example):
            image = example['image']
            return image
        
        dataset = ds.map(extract_image)
        
    except Exception:
        raise ValueError("Error loading dataset")
    
    # Resize to 64x64
    def resize_image(image):
        return tf.image.resize(image, target_size, method='bilinear')
    
    dataset = dataset.map(resize_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Deduplication step (as per section 4.1.1)
    # Note: This step requires materializing the dataset, so we do it after resize
    # but before normalization to work with uint8 images for efficient hashing
    if deduplicate:
        # Convert to uint8 for consistent hashing before deduplication
        # Resize outputs float32, so we need to clip and round properly
        def to_uint8(image):
            image = tf.clip_by_value(image, 0.0, 255.0)
            return tf.cast(tf.round(image), tf.uint8)
        
        dataset = dataset.map(to_uint8, num_parallel_calls=tf.data.AUTOTUNE)
        # Deduplicate
        dataset = _deduplicate_dataset(dataset, target_size)
        # Convert back to float32 for normalization
        dataset = dataset.map(lambda x: tf.cast(x, tf.float32), num_parallel_calls=tf.data.AUTOTUNE)
    
    # Normalize to [-1, 1]
    def normalize_image(image):
        image = tf.cast(image, tf.float32)
        return normalize_to_tanh_range(image)
    
    dataset = dataset.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and batch
    if shuffle:
        if shuffle_buffer is None:
            shuffle_buffer = 10000 
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset










# Task3 part
# use the miniimagenet dataset with image size 64*64

### ImageNet-1k Loader
## Output: 32×32 with min-resize + center crop

def imagenet_min_resize_center_crop(image, target_size=(32, 32)):
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    
    # Min-resize: preserve aspect ratio, resize so min dimension = 32
    min_dim = tf.minimum(h, w)
    scale = target_size[0] / min_dim
    
    new_h = tf.cast(h * scale, tf.int32)
    new_w = tf.cast(w * scale, tf.int32)
    
    resized = tf.image.resize(image, [new_h, new_w], method='bilinear')
    
    crop_size = target_size[0]
    offset_h = (new_h - crop_size) // 2
    offset_w = (new_w - crop_size) // 2
    
    cropped = tf.image.crop_to_bounding_box(
        resized, offset_h, offset_w, crop_size, crop_size
    )
    
    return cropped


# def load_imagenet(batch_size=128, shuffle_buffer=None, target_size=(32, 32), 
#                   subset='train', shuffle=True):
#     # Pipeline: Load → Min-resize (preserve aspect ratio) → Center crop to 32×32 → Normalize → Batch of 128
#     assert batch_size == 128, "Batch size must be 128 per specification"
#     assert target_size == (32, 32), "ImageNet output must be 32x32 per specification"
    
#     try:
#         ds = tfds.load('imagenet_resized/32x32', split=subset, as_supervised=False, 
#                       download=False)
        
#         def extract_image(example):
#             image = example['image']
#             return image
        
#         dataset = ds.map(extract_image)
        
#     except Exception:
#         raise ValueError("Error loading dataset")
    
#     # Apply min-resize and center crop
#     dataset = dataset.map(
#         lambda x: imagenet_min_resize_center_crop(x, target_size),
#         num_parallel_calls=tf.data.AUTOTUNE
#     )
    
#     # Normalize to [-1, 1]
#     def normalize_image(image):
#         image = tf.cast(image, tf.float32)
#         return normalize_to_tanh_range(image)
    
#     dataset = dataset.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    
#     # Shuffle and batch
#     if shuffle:
#         if shuffle_buffer is None:
#             shuffle_buffer = 10000 
#         dataset = dataset.shuffle(shuffle_buffer)
#     dataset = dataset.batch(batch_size, drop_remainder=True)
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
#     return dataset




'''
Date: <09/11/2025>
Written by: <Ziyi Zhao> <zz3459@columbia.edu>

The code in this file was fully implemented by the student.
It has not been generated by AI tools, and it has not been copied 
from any external resource.
'''


def load_local_imagenet(data_dir, batch_size=128, shuffle_buffer=10000, 
                        target_size=(32, 32), shuffle=True):
    """
    Modified loader: Specifically for loading local ImageNet folders.
    Pipeline: Load -> Resize -> Center Crop -> Normalize -> Batch
    """
    print(f"Loading local dataset from: {data_dir}")

    try:
        # Set image_size to (64, 64)
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            label_mode=None,       # Unsupervised learning
            image_size=(64, 64),   
            batch_size=batch_size,      
            shuffle=shuffle
        )
    except Exception as e:
        raise ValueError(f"Error reading directory: {e}")

    # We must unbatch to process images individually to meet the resize function.
    dataset = dataset.unbatch()
    
    # tf.image.resize will work correctly now that input has a fixed shape
    dataset = dataset.map(
        lambda x: imagenet_min_resize_center_crop(x, target_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Normalize to [-1, 1]
    def normalize_image(image):
        image = tf.cast(image, tf.float32)
        return normalize_to_tanh_range(image)
    
    dataset = dataset.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and Batch
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset








'''
Date: <09/11/2025>
Written by: <Ziyi Zhao> <zz3459@columbia.edu>

The code in this file was fully implemented by the student.
It has not been generated by AI tools, and it has not been copied 
from any external resource.
'''





# Task3 part
# use cifar10 dataset to test accuracy
# Here we use simple coding because cifar10 dataset with original size 32,32 and we do only need to normalize and flatten labels
def load_cifar10(target_size=(32, 32)):
    """
    Load CIFAR-10 dataset specifically for Section 5.1 Evaluation.
    Returns normalized Numpy arrays with labels, ready for SVM classification.
    
    Returns:
        (x_train, y_train), (x_test, y_test)
        - x data: normalized to [-1, 1] for Tanh compatibility
        - y data: flattened 1D arrays
    """
    import tensorflow as tf
    import numpy as np

    print("Loading CIFAR-10 for Supervised Linear Classification (Evaluation only)...")
    
    # Load Data with Labels
    # [cite_start]We need labels (y) to train the Linear SVM classifier [cite: 161]
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize to [-1, 1]
    # The Discriminator expects inputs in [-1, 1] range.
    x_train = (x_train / 127.5) - 1.0
    x_test = (x_test / 127.5) - 1.0

    # Flatten Labels
    # Scikit-learn expects 1D arrays for labels, not (N, 1)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    print(f"Data Loaded: Train {x_train.shape}, Test {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)
















### Faces Dataset Loader (CelebA)
## Output: 64×64 (CelebA images are already aligned/cropped faces)

def load_faces_dataset(data_dir, batch_size=128, shuffle_buffer=None, 
                       target_size=(64, 64), shuffle=True):

    # Simplified pipeline for CelebA: Load image → Resize to 64×64 → Normalize → Batch of 128
    assert batch_size == 128, "Batch size must be 128 per specification"
    assert target_size == (64, 64), "Faces output must be 64x64 per specification"
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError("Error loading dataset")
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(data_path.glob(ext)))
        image_files.extend(list(data_path.glob(ext.upper())))
    
    if len(image_files) == 0:
        raise ValueError("Error loading dataset")
    
    # Create dataset from file paths
    file_paths = [str(f) for f in image_files]
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    
    def load_and_preprocess_image(file_path):
        # Read image
        image = tf.io.read_file(file_path)
        # Decode image (CelebA is JPEG)
        image = tf.image.decode_jpeg(image, channels=3)
        # Resize to 64x64
        image = tf.image.resize(image, target_size, method='bilinear')
        # Convert to uint8 (resize outputs float32)
        image = tf.cast(image, tf.uint8)
        return image
    
    # Load and preprocess images
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Normalize to [-1, 1]
    def normalize_image(image):
        image = tf.cast(image, tf.float32)
        return normalize_to_tanh_range(image)
    
    dataset = dataset.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and batch
    if shuffle:
        if shuffle_buffer is None:
            shuffle_buffer = 10000 
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


### Evaluation Dataset Loaders
## These datasets are for evaluation only, not for training DCGAN (Important!)

def load_cifar10(batch_size=128, shuffle_buffer=None, target_size=(32, 32), shuffle=True):
    # Load and preprocess CIFAR-10 dataset for evaluation (feature extraction)
    # Standard CIFAR-10, NOT for training. Used for Table 1 results. Shape: (32, 32, 3)
    assert batch_size == 128, "Batch size must be 128 per specification"
    assert target_size == (32, 32), "CIFAR-10 output must be 32x32 per specification"
    
    # Load CIFAR-10 dataset
    (train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    
    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    
    def normalize_image(image):
        image = tf.cast(image, tf.float32)
        return normalize_to_tanh_range(image)
    
    dataset = dataset.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and batch
    if shuffle:
        if shuffle_buffer is None:
            shuffle_buffer = 10000 
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def load_svhn(batch_size=128, shuffle_buffer=None, target_size=(32, 32), 
              split='train', num_labeled=None, shuffle=True):
    # Load and preprocess SVHN dataset for semi-supervised evaluation
    # Paper spec: Validation 10k from non-extra set, Labeled samples 1k uniformly distributed
    assert batch_size == 128, "Batch size must be 128 per specification"
    assert target_size == (32, 32), "SVHN output must be 32x32 per specification"
    
    # Load SVHN dataset using tensorflow_datasets
    try:
        if split == 'validation':
            # Take 10,000 from non-extra set (train split, first 10k)
            ds = tfds.load('svhn_cropped', split='train[:10000]', as_supervised=True)
        elif split == 'train':
            # Full training set (non-extra)
            ds = tfds.load('svhn_cropped', split='train', as_supervised=True)
        elif split == 'test':
            ds = tfds.load('svhn_cropped', split='test', as_supervised=True)
        else:
            raise ValueError("Error loading dataset")
        
        # Extract images and labels
        def extract_image_label(example):
            image, label = example
            return image, label
        
        dataset = ds.map(extract_image_label)
        
        if num_labeled is not None:
            # Group by label and sample uniformly
            # This requires collecting all data first (for small labeled sets)
            images_list = []
            labels_list = []
            
            for image, label in dataset:
                images_list.append(image.numpy() if hasattr(image, 'numpy') else image)
                labels_list.append(label.numpy() if hasattr(label, 'numpy') else label)
            
            images_array = np.array(images_list)
            labels_array = np.array(labels_list)
            
            # Sample uniformly across classes (10 classes for SVHN: digits 0-9)
            num_classes = 10
            samples_per_class = num_labeled // num_classes
            selected_indices = []
            
            for class_id in range(num_classes):
                class_indices = np.where(labels_array == class_id)[0]
                if len(class_indices) > 0:
                    
                    np.random.seed(42)  
                    selected = np.random.choice(
                        class_indices, 
                        size=min(samples_per_class, len(class_indices)),
                        replace=False
                    )
                    selected_indices.extend(selected)
            
            
            selected_images = images_array[selected_indices]
            dataset = tf.data.Dataset.from_tensor_slices(selected_images)
        else:
            # Extract only images (no labels needed for unsupervised training)
            dataset = dataset.map(lambda img, lbl: img)
        
    except Exception:
        raise ValueError("Error loading dataset")
    
    
    def normalize_image(image):
        image = tf.cast(image, tf.float32)
        return normalize_to_tanh_range(image)
    
    dataset = dataset.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and batch
    if shuffle:
        if shuffle_buffer is None:
            shuffle_buffer = 10000 
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def load_mnist(batch_size=128, shuffle_buffer=None, target_size=(28, 28), shuffle=True):
    # Load and preprocess MNIST dataset for conditional GAN evaluation
    # Converts grayscale to RGB (3 channels) and normalizes to [-1, 1]
    assert batch_size == 128, "Batch size must be 128 per specification"
    
    # Load MNIST dataset
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    
    # Preprocess: Convert to 3 channels and normalize
    def preprocess_mnist(image):
        # Expand to 3 channels (grayscale to RGB)
        image = tf.expand_dims(image, axis=-1)
        image = tf.repeat(image, 3, axis=-1)
        # Normalize to [-1, 1]
        image = tf.cast(image, tf.float32)
        return normalize_to_tanh_range(image)
    
    dataset = dataset.map(preprocess_mnist, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and batch
    if shuffle:
        if shuffle_buffer is None:
            shuffle_buffer = 10000 
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


### Convenience Function

def get_dataset(dataset_name='cifar10', **kwargs):
    # Convenience function to get dataset by name
    # Training: 'lsun', 'imagenet', 'faces'
    # Evaluation: 'cifar10', 'svhn', 'mnist'
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower == 'lsun':
        return load_lsun(**kwargs)
    elif dataset_name_lower == 'imagenet':
        return load_imagenet(**kwargs)
    elif dataset_name_lower == 'faces':
        if 'data_dir' not in kwargs:
            raise ValueError("Error loading dataset")
        return load_faces_dataset(**kwargs)
    elif dataset_name_lower == 'cifar10':
        return load_cifar10(**kwargs)
    elif dataset_name_lower == 'svhn':
        return load_svhn(**kwargs)
    elif dataset_name_lower == 'mnist':
        return load_mnist(**kwargs)
    else:
        raise ValueError("Error loading dataset")


if __name__ == "__main__":
    # Test CIFAR-10 loading
    dataset = load_cifar10(batch_size=128)
