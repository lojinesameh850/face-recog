import os
import numpy as np
from PIL import Image

def load_and_preprocess_orl(dataset_path = "att-database-of-faces.zip"):
    D = []
    y = []
    
    for subject_id in range(1, 41):
        subject_dir = os.path.join(dataset_path, f's{subject_id}')
        
        for img_idx in range(1, 11):
            img_path = os.path.join(subject_dir, f'{img_idx}.pgm')
            img = Image.open(img_path).convert('L')
            img_vector = np.array(img).flatten()
            D.append(img_vector)
            y.append(subject_id)
            
    # Stack into a single Data Matrix D (400x10304) and label vector y 
    D = np.array(D)
    y = np.array(y)
    
    print(f"Data Matrix D shape: {D.shape}")
    print(f"Label vector y shape: {y.shape}")
    
    # Split the Dataset into Training and Test sets: odd for training, even for testing 
    D_train = D[0::2]; D_test = D[1::2]
    y_train = y[0::2]; y_test = y[1::2]
    
    os.makedirs('processed_data', exist_ok=True)
    np.save('processed_data/D_train.npy', D_train)
    np.save('processed_data/y_train.npy', y_train)
    np.save('processed_data/D_test.npy', D_test)
    np.save('processed_data/y_test.npy', y_test)
    
    print("Preprocessing complete. Train and test sets saved to 'processed_data/' directory.")

if __name__ == "__main__":
    dataset_directory = "att-database-of-faces" 
    load_and_preprocess_orl(dataset_directory)