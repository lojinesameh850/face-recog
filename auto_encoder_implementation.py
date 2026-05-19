import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Autoencoder, self).__init__()
        
        encoder_layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            current_dim = h_dim
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        current_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            current_dim = h_dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        decoder_layers.append(nn.Sigmoid()) 
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
def train_autoencoder(X_train, hidden_dims, latent_dim, epochs=50, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim, hidden_dims, latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    dataset = TensorDataset(torch.FloatTensor(X_train))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            x_batch = batch[0].to(device)
            
            optimizer.zero_grad()
            reconstructed, _ = model(x_batch)
            loss = criterion(reconstructed, x_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
            
    return model

def get_latent_projections(model, X):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X).to(device)
        _, latent = model(x_tensor)
        
    return latent.cpu().numpy()

def visualize_sample_faces(models_dict, D_train, X_train_scaled, latent_dims, config_name, sample_idx=0):
    img_shape = (112, 92)
    os.makedirs('auto_encoder_model', exist_ok=True)

    sample_indices = [sample_idx, sample_idx + 1, sample_idx + 2]
    
    plt.figure(figsize=(15, 5 * len(sample_indices)))

    for row, idx in enumerate(sample_indices):
        original_face = D_train[idx].reshape(img_shape)
        plt.subplot(len(sample_indices), len(latent_dims) + 1, row * (len(latent_dims) + 1) + 1)
        plt.imshow(original_face, cmap='gray')
        plt.title(f"Original Face {idx}")
        plt.axis('off')

        np.save(f'auto_encoder_model/original_face_{idx}.npy', D_train[idx])
        plt.imsave(f'auto_encoder_model/original_face_{idx}.png', original_face, cmap='gray')

        for i, l_dim in enumerate(latent_dims):
            model = models_dict[l_dim]
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                x_tensor = torch.FloatTensor(X_train_scaled[idx]).unsqueeze(0).to(device)
                reconstructed_tensor, _ = model(x_tensor)
                reconstructed_flat = reconstructed_tensor.squeeze().cpu().numpy()
            
            reconstructed_face = reconstructed_flat * 255.0
            recon_img = reconstructed_face.reshape(img_shape)

            plt.subplot(len(sample_indices), len(latent_dims) + 1, row * (len(latent_dims) + 1) + i + 2)
            plt.imshow(recon_img, cmap='gray')
            plt.title(f"Latent Dim: {l_dim}")
            plt.axis('off')

            np.save(f'auto_encoder_model/reconstructed_face_{idx}_ld_{l_dim}_{config_name}.npy', reconstructed_face)
            plt.imsave(f'auto_encoder_model/reconstructed_face_{idx}_ld_{l_dim}_{config_name}.png', recon_img, cmap='gray')

    plt.tight_layout()
    figure_path = f'auto_encoder_model/reconstructed_samples_{config_name}.png'
    plt.savefig(figure_path)
    plt.close()
    print(f"Saved reconstructed face images and arrays to 'auto_encoder_model/'.")

if __name__ == "__main__":
    print("Loading data...")
    D_train = np.load("processed_data/D_train.npy")
    D_test = np.load("processed_data/D_test.npy")

    X_train = D_train / 255.0
    X_test = D_test / 255.0

    LATENT_DIMS = [50, 100, 200]
    
    HIDDEN_CONFIGS = [
        [512, 128],          
        [1024, 256, 64],     
        [256, 64]            
    ]
    
    LEARNING_RATES = [1e-3]
    EPOCHS = 50

    os.makedirs("auto_encoder_projected_data", exist_ok=True)
    os.makedirs("auto_encoder_model", exist_ok=True)

    for hidden_dims in HIDDEN_CONFIGS:
        models_for_viz = {}
        config_base_name = f"hd_{'-'.join(map(str, hidden_dims))}_lr_{LEARNING_RATES[0]}"
        
        for latent_dim in LATENT_DIMS:
            for lr in LEARNING_RATES:
                config_str = f"hd_{'-'.join(map(str, hidden_dims))}_ld_{latent_dim}_lr_{lr}"
                print(f"\nTraining Config: {config_str}")
                
                ae_model = train_autoencoder(
                    X_train, 
                    hidden_dims=hidden_dims, 
                    latent_dim=latent_dim, 
                    epochs=EPOCHS, 
                    lr=lr
                )

                print("  Generating projections...")
                X_train_reduced = get_latent_projections(ae_model, X_train)
                X_test_reduced = get_latent_projections(ae_model, X_test)

                np.save(f"auto_encoder_projected_data/train_ae_{config_str}.npy", X_train_reduced)
                np.save(f"auto_encoder_projected_data/test_ae_{config_str}.npy", X_test_reduced)

                torch.save(ae_model.state_dict(), f"auto_encoder_model/autoencoder_{config_str}.pth")
                
                models_for_viz[latent_dim] = ae_model
                
        visualize_sample_faces(models_for_viz, D_train, X_train, LATENT_DIMS, config_base_name, sample_idx=5)
                
    print("\n All hyperparameter combinations trained, saved, and visualized successfully!")