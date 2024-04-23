# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 14:53:29 2024

@author: arwilzman
"""
import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import gc
begin_start = time.time()
# Argument Parser
parser = argparse.ArgumentParser(description="Hyperparameters for training")
parser.add_argument('--load_model', type=str, default='',
                    help='Path to pre-trained model (default: None)')
parser.add_argument('--hidden', type=int, default=512,
                    help='Hidden size for training (default: 512)')
parser.add_argument('--latent', type=int, default=256,
                    help='Latent size of encoder (default: 256)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for training (default: 32)')
parser.add_argument('--lr', type=float, nargs='+', default=[1e-2, 1e-3, 1e-4],
                    help='Learning rates to schedule (default: [1e-2, 1e-3, 1e-4])')
parser.add_argument('--wtdecay', type=float, default=1e-6,
                    help='Weight decay (default: 1e-6)')
parser.add_argument('--noise', type=int, default=3,
                    help='Diffusion noise level (1-10) (default: 3)')
parser.add_argument('-e','--epochs', type=int, default=10,
                    help='Number of epochs (default: 10)')
parser.add_argument('--print_interval', type=int, default=None,
                    help='Interval for printing progress (default: epochs//10)')
parser.add_argument('-p','--plot', action='store_true',
                    help='Flag to plot the results (default: False)')
args = parser.parse_args()
# Load data
The_Data = pd.read_pickle('JS_04_23_24.pkl')

# Remove unwanted columns
col_remover = ['IMU','pX','pY','pZ','Sex','Tibia Stiffness','MassN','time']
for col in col_remover:
    The_Data = The_Data.loc[:, ~The_Data.columns.str.contains(col)]

data = pd.read_csv('Jump_Study_Data_04_19_24.csv')

# Columns to match
match_cols = ['ID', 'Landing Limbs', 'Trial No']

# Create a new column in dfs that combines the matching columns
The_Data['combined_id'] = The_Data[match_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
data['combined_id'] = data[match_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

# Prepare data subsets
Data_subsets = {}
for value in data['combined_id'].unique():
    Data_subset = The_Data[The_Data['combined_id'] == value].copy()
    Data_subset = Data_subset.drop(columns=['ID', 'Trial No', 'combined_id'])
    Data_subsets[value] = np.array(Data_subset)  # Convert DataFrame to NumPy array


#%%
def train_autoencoder(training_inputs, network, epochs, learning_rate, wtdecay,
                      batch_size, loss_function, print_interval, device):
    
    train_loader = DataLoader(training_inputs, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=wtdecay)
    
    track_losses = []
    start = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_idx, X in enumerate(train_loader):
            X = X.to(device)
            X = torch.nan_to_num(X, nan=0.0)
            # Forward pass
            output, latent, _ = network(X)
            
            # Compute loss
            loss = loss_function(output, X)*1e-5
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            for param in network.parameters():
                nn.utils.clip_grad_norm_(param, max_norm=1.0)
                
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Compute average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        track_losses.append(avg_epoch_loss)
        
        if epoch % print_interval == 0:
            print(f"Epoch [{epoch}/{epochs}], Avg Loss: {avg_epoch_loss:.4f}, Time: {time.time() - start:.2f} sec")
        
    return network, track_losses

def train_vae(training_inputs, network, epochs, learning_rate, wtdecay,
              batch_size, loss_function, print_interval, device):
    
    train_loader = DataLoader(training_inputs, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=wtdecay)
    
    track_losses = []
    start = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_idx, X in enumerate(train_loader):
            X = X.to(device)
            X = torch.nan_to_num(X, nan=0.0)
            # Forward pass
            output, latent, _ = network(X)
            
            mean = latent.mean(dim=1).squeeze(0)
            std = latent.std(dim=1, unbiased=False).squeeze(0)
            
            # Create a normal distribution for each feature
            normal_dist = torch.distributions.Normal(mean, std)

            # Generate samples using rsample for each distribution
            samples = [normal_dist.rsample() for i in range(latent.shape[1])]
            
            # Stack the samples along dimension 2
            z = torch.stack(samples).reshape(-1,1,latent.shape[1])
            
            std_normal = torch.distributions.Normal(0,1)
            
            loss_kl = torch.distributions.kl.kl_divergence(normal_dist, std_normal).mean()
            
            # Compute loss
            loss = loss_function(output, X)*1e-5 + loss_kl
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            for param in network.parameters():
                nn.utils.clip_grad_norm_(param, max_norm=1.0)
                
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Compute average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        track_losses.append(avg_epoch_loss)
        
        if epoch % print_interval == 0:
            print(f"Epoch [{epoch}/{epochs}], Avg Loss: {avg_epoch_loss:.4f}, Time: {time.time() - start:.2f} sec")
        
    return network, track_losses

def train_diff(training_inputs, network, epochs, learning_rate, wtdecay,
               batch_size, loss_function, print_interval, noise_level, device):
    
    train_loader = DataLoader(training_inputs, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=wtdecay)
    
    track_losses = []
    start = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_idx, X in enumerate(train_loader):
            X = X.to(device)
            X = torch.nan_to_num(X, nan=0.0)
            # Forward pass
            latent = network.encode(X)
            #diffuse
            lamb_t = max(min(noise_level/10,1),0.05)
            sig_t = 1-lamb_t
            noise = noise_level * torch.randn_like(latent).to(device)
            noisy = sig_t * latent + noise
            output = network.decode(latent,X.shape[1])
            
            # Compute loss
            loss = loss_function(output, X)*1e-5
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            for param in network.parameters():
                nn.utils.clip_grad_norm_(param, max_norm=1.0)
                
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Compute average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        track_losses.append(avg_epoch_loss)
        
        if epoch % print_interval == 0:
            print(f"Epoch [{epoch}/{epochs}], Avg Loss: {avg_epoch_loss:.4f}, Time: {time.time() - start:.2f} sec")
        
    return network, track_losses

def train_GD(training_inputs, network, epochs, learning_rate, decay,
             batch_size, loss_function, print_interval,device): 
    train_loader = DataLoader(training_inputs, batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    
    optimizerG = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=decay)
    optimizerD = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=decay)
    G_losses = np.zeros(1)
    D_losses = np.zeros(1)
    network = network.to(device)
    start = time.time()
    print('timer started')
    for epoch in range(1, epochs+1):
        for batch_idx, X in enumerate(train_loader):
            X = X.to(device)
            X = torch.nan_to_num(X, nan=0.0)
            b_size = len(X)
            label = torch.full((b_size,),1,dtype=torch.float,device=device)
            
            # train with real
            _,b,output = network(X)
            output = output.squeeze(1)
            # find the loss
            lossD_real = loss_function(output, label)
            # compute the gradient and update the network parameters
            optimizerD.zero_grad()
            lossD_real.backward()
            optimizerD.step()
            D_x = output.mean().item() # 1 if discriminator is doing well
            
            noise = torch.randn_like(b, device=device)
            
            fake = network.decode(noise,b.shape[1])
            
            label.fill_(0)
            _,_,output = network(fake)
            output = output.squeeze(1)
            
            D_G_z1 = output.mean().item() # 0 if discriminator is doing well, 
            
            optimizerD.zero_grad()
            lossD_fake = loss_function(output, label)# closer to 1, generator is foolin'
            
            lossD_fake.backward()
            optimizerD.step()
            _,_,output = network(fake.detach()) 
            output = output.squeeze(1)
            
            # 0 if discriminator is doing well after training step
            D_G_z2 = output.mean().item() # closer to 1, generator still foolin'
            
            #update G
            label.fill_(1)
            lossG = loss_function(output, label)
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
        # housekeeping - keep track of our losses and print them as we go
        lossD = lossD_real + lossD_fake
        G_losses = np.append(G_losses,lossG.item())
        D_losses = np.append(D_losses,lossD.item())
        if epoch % print_interval == 0:
            print('[%d/%d][%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, epochs, time.time()-start,
                       lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
    return network,G_losses,D_losses

class RNNAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNNAutoencoder, self).__init__()
        self.activate = nn.LeakyReLU()
        # Encoder
        self.encoder_rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        self.dnet = nn.Sequential(nn.Linear(latent_dim,latent_dim//2),
                                  self.activate,
                                  nn.Linear(latent_dim//2,4),
                                  self.activate,
                                  nn.Linear(4,1),
                                  nn.Sigmoid())
        # Latent space
        self.latent_space = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        lengths = (x.sum(dim=-1) != 0).sum(dim=1)
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        _, hidden = self.encoder_rnn(packed_x)
        
        # Flatten and project to latent space
        latent = self.activate(self.latent_space(hidden.squeeze(0)))

        return latent

    def decode(self, latent, length):
        # Repeat latent vector for sequence length
        latent = latent.unsqueeze(1).repeat(1, length, 1)
        x, _ = self.decoder_rnn(latent)
        
        # Output layer
        reconstructed_data = self.activate(self.output_layer(x.data))

        return reconstructed_data
    
        
    def forward(self, x):
        # Encode
        latent = self.encode(x)
        discrim = self.dnet(latent)
        # Decode
        reconstructed_data = self.decode(latent, x.shape[1])

        return reconstructed_data, latent, discrim


def collate_fn(batch):
    # Sort sequences by length
    batch.sort(key=lambda x: len(x), reverse=True)
    
    # Pad sequences to have the same length
    lengths = [len(x) for x in batch]
    max_len = max(lengths)
    padded_batch = [torch.cat((x, torch.zeros(max_len - len(x), x.shape[1]))) for x in batch]
    
    # Convert to tensor
    padded_batch = torch.stack(padded_batch)
    
    return padded_batch

if torch.cuda.is_available():
    print('CUDA available')
    print(torch.cuda.get_device_name(0))
else:
    print('CUDA *not* available')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert data subsets to list of tensors
training_inputs = [torch.tensor(sample, dtype=torch.float32) for sample in Data_subsets.values()]

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model parameters
torch.manual_seed(42)
input_dim = training_inputs[0].shape[1]
hidden_dim = args.hidden
latent_dim = args.latent

# Initialize model
network = RNNAutoencoder(input_dim, hidden_dim, latent_dim).to(device)

# Load model if specified
if args.load_model:
    network.load_state_dict(torch.load(args.load_model))

# Initialize losses dictionary
losses_dict = {}

# Train the model for each learning rate
for learning_rate in args.lr:
    print(f'Starting training with learning rate {learning_rate}')
    
    ae_network, ae_losses = train_autoencoder(training_inputs, network, args.epochs, learning_rate,
                                              args.wtdecay, args.batch_size, nn.MSELoss(),
                                              args.print_interval or args.epochs//10, device)
    torch.cuda.empty_cache()
    vae_network, vae_losses = train_vae(training_inputs, ae_network, args.epochs, learning_rate,
                                        args.wtdecay, args.batch_size, nn.MSELoss(),
                                        args.print_interval or args.epochs//10, device)
    torch.cuda.empty_cache()
    diff_network, diff_losses = train_diff(training_inputs, ae_network, args.epochs, learning_rate,
                                           args.wtdecay, args.batch_size, nn.MSELoss(),
                                           args.print_interval or args.epochs//10, 3, device)
    torch.cuda.empty_cache()
    GAN_network, g_losses, d_losses = train_GD(training_inputs, diff_network, args.epochs, learning_rate,
                                               args.wtdecay, args.batch_size, nn.MSELoss(),
                                               args.print_interval or args.epochs//10, device)
    torch.cuda.empty_cache()
    gc.collect()
    
    network=GAN_network
    
    # Store losses in the dictionary
    losses_dict[learning_rate] = {
        'ae': ae_losses,
        'vae': vae_losses,
        'diff': diff_losses,
        'g': g_losses,
        'd': d_losses
    }
torch.save(ae_network.state_dict(), f'ae_network.pth')
torch.save(vae_network.state_dict(), f'vae_network.pth')
torch.save(diff_network.state_dict(), f'diff_network.pth')
torch.save(GAN_network.state_dict(), f'GAN_network.pth')

loss_df = pd.DataFrame.from_dict({(i, j): losses_dict[i][j]
                                  for i in losses_dict.keys()
                                  for j in losses_dict[i].keys()},
                                 orient='index')
loss_df.to_csv('losses.csv')

# Plotting loss curves if the plot flag is set
if args.plot:
    import matplotlib.pyplot as plt
    num_loss_types = len(losses_dict[args.lr[0]])  # Assuming all learning rates have the same loss types
    num_learning_rates = len(args.lr)

    fig, axs = plt.subplots(num_loss_types, num_learning_rates, figsize=(20, 15))

    # Calculate separate minimum and maximum loss values for each training type across all learning rates
    type_min_losses = {loss_type: min([min(losses_dict[lr][loss_type]) for lr in args.lr])
                       for loss_type in losses_dict[args.lr[0]]}
    type_max_losses = {loss_type: max([max(losses_dict[lr][loss_type]) for lr in args.lr])
                       for loss_type in losses_dict[args.lr[0]]}

    # Plotting loss curves for each learning rate
    for j, loss_type in enumerate(losses_dict[args.lr[0]].keys()):
        for i, learning_rate in enumerate(args.lr):
            ax = axs[j, i]
            losses = losses_dict[learning_rate][loss_type]
            ax.plot(losses, label=f'LR: {learning_rate}', color='blue')
            ax.set_title(f'{loss_type.capitalize()} Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)

            # Set y-axis limits tailored to each training type
            ax.set_ylim(type_min_losses[loss_type], type_max_losses[loss_type])

    plt.tight_layout()
    plt.show()

total_time = time.time() - begin_start

print(f'Total time: {total_time}')

#%%
# Function to plot samples
def plot_samples(original, generated, title):
    plt.figure(figsize=(10, 5))
    
    # Plot original sample
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(original.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    
    # Plot generated sample
    plt.subplot(1, 2, 2)
    plt.title('Generated from Noise')
    plt.imshow(generated.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.show()

# Load a single training sample
training_sample = training_inputs[0].unsqueeze(0).to(device)
latent_space = network.encode(training_sample)
# Generate noise
noise = torch.randn_like(latent_space)

# Generate samples using each trained network from noise
with torch.no_grad():
    # Autoencoder
    ae_generated = ae_network.decode(noise,99)
    
    # VAE
    vae_generated = vae_network.decode(noise,99)
    
    # Differential network
    diff_generated = diff_network.decode(noise,99)
    
    # GAN
    gan_generated = GAN_network.decode(noise,99)

# Plot samples
plot_samples(training_sample, ae_generated, 'Autoencoder')
plot_samples(training_sample, vae_generated, 'VAE')
plot_samples(training_sample, diff_generated, 'Differential Network')
plot_samples(training_sample, gan_generated, 'GAN')