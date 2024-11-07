import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def create_folders(base_path):
    """Create necessary folders for dataset organization"""
    # Create main folders
    folders = ['train', 'val', 'test']
    
    # List of skin lesion classes
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    # Create folder structure
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Create class subfolders
        for class_name in classes:
            class_path = os.path.join(folder_path, class_name)
            if not os.path.exists(class_path):
                os.makedirs(class_path)

def organize_dataset(metadata_path, image_dir, processed_dir):
    """Organize dataset into train, validation, and test sets"""
    # Read metadata
    df = pd.read_csv(metadata_path)
    
    # Create folders
    create_folders(processed_dir)
    
    # Split dataset
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['dx'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dx'])
    
    # Function to copy images
    def copy_images(dataframe, split_type):
        for index, row in dataframe.iterrows():
            # Source image path
            src = os.path.join(image_dir, row['image_id'] + '.jpg')
            
            # Destination path
            dst = os.path.join(processed_dir, split_type, row['dx'], row['image_id'] + '.jpg')
            
            # Copy image
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"Warning: {src} not found")
    
    # Copy images to respective folders
    print("Copying training images...")
    copy_images(train_df, 'train')
    
    print("Copying validation images...")
    copy_images(val_df, 'val')
    
    print("Copying test images...")
    copy_images(test_df, 'test')
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

def main():
    # Define paths
    # Change these paths to match your actual dataset location
    METADATA_PATH = r"C:\Users\Asus\Desktop\skin_cancer_project\data\raw\HAM10000_metadata.csv"  # Change this
    IMAGE_DIR = r"C:\Users\Asus\Desktop\skin_cancer_project\data\raw\HAM1000_images"  # Change this
    
    # This is where the processed dataset will be saved (within your project folder)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    
    # Organize dataset
    organize_dataset(METADATA_PATH, IMAGE_DIR, PROCESSED_DATA_DIR)

if __name__ == "__main__":
    main()