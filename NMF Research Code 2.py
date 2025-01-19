# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
import pandas as pd
import seaborn as sns

# %%
warnings.filterwarnings("ignore", category=ConvergenceWarning)
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = data.data 
images = data.images
n_samples, n_features = X.shape

print(f"Number of samples: {n_samples}")
print(f"Number of features (pixels per image): {n_features}")


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

n_components = 40

init_methods = ['random', 'nndsvd', 'nndsvda', 'nndsvdar']

results = {}

init_methods_list = []
loss_frobenius_list = []

for init_method in init_methods:
    print(f"Running NMF with init method: '{init_method}'")
    
    model = NMF(
        n_components=n_components,
        init=init_method,
        random_state=42,
        max_iter=500,
        solver='cd',  
        beta_loss='frobenius',
        tol=1e-4
    )

    W = model.fit_transform(X_scaled)
    H = model.components_
    
    X_reconstructed = np.dot(W, H)
    
    loss_frobenius = np.linalg.norm(X_scaled - X_reconstructed, 'fro')
    
    results[init_method] = {
        'model': model,
        'W': W,
        'H': H,
        'X_reconstructed': X_reconstructed,
        'loss_frobenius': loss_frobenius
    }
    
    init_methods_list.append(init_method)
    loss_frobenius_list.append(loss_frobenius)


loss_df = pd.DataFrame({
    'Initialization Method': init_methods_list,
    'Frobenius Norm Loss': loss_frobenius_list
})

print("\nFrobenius Norm Loss for Each Initialization Method:")
print(loss_df)

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.barplot(x='Initialization Method', y='Frobenius Norm Loss', data=loss_df, palette='viridis')
plt.title('Frobenius Norm Loss by Initialization Method')
plt.ylabel('Frobenius Norm Loss')
plt.xlabel('Initialization Method')
plt.tight_layout()
plt.show()

# display basis images
def plot_images(images, titles, h, w, n_row=2, n_col=8, suptitle="Images"):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        if i >= len(images):
            break
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.suptitle(suptitle)
    plt.show()

# display original and reconstructed images side by side
def plot_reconstructed(original, reconstructed, n_images=10, suptitle="Original vs Reconstructed"):
    plt.figure(figsize=(20, 4))
    for i in range(n_images):
        # Original Image
        ax = plt.subplot(2, n_images, i + 1)
        plt.imshow(original[i].reshape(64, 64), cmap='gray')
        if i == 0:
            plt.ylabel("Original", size=14)
        plt.title(f"Image {i+1}", size=12)
        plt.xticks([])
        plt.yticks([])
        
        # Reconstructed Image
        ax = plt.subplot(2, n_images, i + 1 + n_images)
        plt.imshow(reconstructed[i].reshape(64, 64), cmap='gray')
        if i == 0:
            plt.ylabel("Reconstructed", size=14)
        plt.title(f"Image {i+1}", size=12)
        plt.xticks([])
        plt.yticks([])
    plt.suptitle(suptitle)
    plt.show()

for init_method in init_methods:
    H = results[init_method]['H']
    X_reconstructed = results[init_method]['X_reconstructed']
    
    # Plot the basis images
    titles = [f"Comp {i+1}" for i in range(n_components)]
    plot_images(
        H, 
        titles, 
        h=64, 
        w=64, 
        n_row= n_components//10, 
        n_col=10, 
        suptitle=f"NMF Basis Images ({init_method})"
    )
    
    # Plot original vs reconstructed images
    plot_reconstructed(
        X_scaled, 
        X_reconstructed, 
        n_images=10, 
        suptitle=f"Original vs Reconstructed Images ({init_method})"
    )


# %%



