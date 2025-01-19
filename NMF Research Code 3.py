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

n_components = 

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
warnings.filterwarnings("ignore", category=ConvergenceWarning)
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = data.data 
images = data.images
n_samples, n_features = X.shape

print(f"Number of samples: {n_samples}")
print(f"Number of features (pixels per image): {n_features}")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the train-test split ratio
test_size = 0.2
random_state = 42 

X_train, X_test, images_train, images_test = train_test_split(
    X_scaled, images, test_size=test_size, random_state=random_state, shuffle=True
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")


n_components = 40  

init_methods = ['random', 'nndsvd', 'nndsvda', 'nndsvdar']

results = {}

for init_method in init_methods:
    
    model = NMF(
        n_components=n_components,
        init=init_method,
        random_state=random_state,
        max_iter=1000,
        solver='cd', 
        beta_loss='frobenius',
        tol=1e-4
    )
    
    W_train = model.fit_transform(X_train)
    H = model.components_
    
    X_train_reconstructed = np.dot(W_train, H)
    
    train_error = np.linalg.norm(X_train - X_train_reconstructed, 'fro')
    
    W_test = model.transform(X_test)
    X_test_reconstructed = np.dot(W_test, H)
    
    test_error = np.linalg.norm(X_test - X_test_reconstructed, 'fro')
    
    print(f"Training Reconstruction Error (Frobenius norm): {train_error:.6f}")
    print(f"Test Reconstruction Error (Frobenius norm): {test_error:.6f}")
    

    results[init_method] = {
        'model': model,
        'W_train': W_train,
        'H': H,
        'X_train_reconstructed': X_train_reconstructed,
        'train_error': train_error,
        'W_test': W_test,
        'X_test_reconstructed': X_test_reconstructed,
        'test_error': test_error
    }

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

# Visualize the results for each initialization method
for init_method in init_methods:
    print(f"\nVisualizing results for init method: '{init_method}'")
    result = results[init_method]
    H = result['H']
    X_test_reconstructed = result['X_test_reconstructed']
    
    # Plot the basis images
    titles = [f"Comp {i+1}" for i in range(n_components)]
    plot_images(
        H, 
        titles, 
        h=64, 
        w=64, 
        n_row=2, 
        n_col=8, 
        suptitle=f"NMF Basis Images ({init_method})"
    )
    
    # Plot original vs reconstructed test images
    plot_reconstructed(
        X_test, 
        X_test_reconstructed, 
        n_images=10, 
        suptitle=f"Original vs Reconstructed Test Images ({init_method})"
    )

# Compare reconstruction errors
print("\nReconstruction Errors for Different Init Methods:")
for init_method in init_methods:
    train_err = results[init_method]['train_error']
    test_err = results[init_method]['test_error']
    print(f" - {init_method}: Train Frobenius Norm = {train_err:.6f}, Test Frobenius Norm = {test_err:.6f}")

# Visualize reconstruction errors as a bar chart
plt.figure(figsize=(10,6))
train_errors = [results[init]['train_error'] for init in init_methods]
test_errors = [results[init]['test_error'] for init in init_methods]

x = np.arange(len(init_methods)) 
width = 0.35 

fig, ax = plt.subplots(figsize=(10,6))
rects1 = ax.bar(x - width/2, train_errors, width, label='Train Frobenius Norm', color='skyblue')
rects2 = ax.bar(x + width/2, test_errors, width, label='Test Frobenius Norm', color='salmon')

ax.set_ylabel('Frobenius Norm')
ax.set_title('Reconstruction Error by Initialization Method')
ax.set_xticks(x)
ax.set_xticklabels(init_methods)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.ylim(0, max(max(train_errors), max(test_errors)) * 1.1)
plt.show();


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress only ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = data.data  # Shape: (400, 4096) where each image is 64x64 pixels
images = data.images
n_samples, n_features = X.shape

print(f"Number of samples: {n_samples}")
print(f"Number of features (pixels per image): {n_features}")

# Scale the data (optional, already between 0 and 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define the train-test split ratio
test_size = 0.2  # 20% for testing
random_state = 42  # For reproducibility

# Split the data
X_train, X_test, images_train, images_test = train_test_split(
    X_scaled, images, test_size=test_size, random_state=random_state, shuffle=True
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# Define the range of n_components to explore
n_components_range = range(5, 100, 5)  

# Define the list of initialization methods to iterate over
init_methods = ['random', 'nndsvd', 'nndsvda', 'nndsvdar']

# Dictionary to store results for each combination of n_components and init_method
results = {}

# Iterate over each n_components value
for n_components in n_components_range:
    print(f"\n=== Evaluating n_components = {n_components} ===")
    # Iterate over each initialization method
    for init_method in init_methods:
        print(f"Running NMF with init method: '{init_method}'")
        # Initialize the NMF model
        model = NMF(
            n_components=n_components,
            init=init_method,
            random_state=random_state,
            max_iter=500,
            solver='cd',  
            beta_loss='frobenius',
            tol=1e-4
        )
        
        # Fit the model to the training data
        W_train = model.fit_transform(X_train)
        H = model.components_
        
        # Reconstruct the training data
        X_train_reconstructed = np.dot(W_train, H)
        
        # Calculate the reconstruction error on the training set using the Frobenius norm
        train_error = np.linalg.norm(X_train - X_train_reconstructed, 'fro')
        
        # Transform and reconstruct the test data
        W_test = model.transform(X_test)
        X_test_reconstructed = np.dot(W_test, H)
        
        # Calculate the reconstruction error on the test set using the Frobenius norm
        test_error = np.linalg.norm(X_test - X_test_reconstructed, 'fro')
        
        print(f"Training Reconstruction Error (Frobenius norm): {train_error:.6f}")
        print(f"Test Reconstruction Error (Frobenius norm): {test_error:.6f}")
        
        # Store the results
        results[(n_components, init_method)] = {
            'model': model,
            'W_train': W_train,
            'H': H,
            'X_train_reconstructed': X_train_reconstructed,
            'train_error': train_error,
            'W_test': W_test,
            'X_test_reconstructed': X_test_reconstructed,
            'test_error': test_error
        }

# Identify the best combination based on the lowest test error
best_combination = None
lowest_test_error = np.inf
lowest_train_error = np.inf

for (n_components, init_method), result in results.items():
    train_err = result['train_error']
    test_err = result['test_error']
    
    # Criteria: Select the combination with the lowest test error
    # In case of ties, prioritize lower train error
    if test_err < lowest_test_error or (test_err == lowest_test_error and train_err < lowest_train_error):
        lowest_test_error = test_err
        lowest_train_error = train_err
        best_combination = (n_components, init_method)

print("\n=== Best Combination ===")
print(f"n_components: {best_combination[0]}")
print(f"init_method: '{best_combination[1]}'")
print(f"Training Reconstruction Error (Frobenius norm): {lowest_train_error:.6f}")
print(f"Test Reconstruction Error (Frobenius norm): {lowest_test_error:.6f}")

# Prepare data for heatmaps
train_errors_matrix = pd.DataFrame(
    index=n_components_range,
    columns=init_methods,
    data=np.nan
)

test_errors_matrix = pd.DataFrame(
    index=n_components_range,
    columns=init_methods,
    data=np.nan
)

for (n_components, init_method), result in results.items():
    train_errors_matrix.loc[n_components, init_method] = result['train_error']
    test_errors_matrix.loc[n_components, init_method] = result['test_error']

# Plot Heatmap for Training Errors
plt.figure(figsize=(10, 6))
sns.heatmap(train_errors_matrix, annot=True, fmt=".4f", cmap='viridis')
plt.title('Training Reconstruction Error (Frobenius Norm)')
plt.ylabel('n_components')
plt.xlabel('Initialization Method')
plt.show()

# Plot Heatmap for Test Errors
plt.figure(figsize=(10, 6))
sns.heatmap(test_errors_matrix, annot=True, fmt=".4f", cmap='viridis')
plt.title('Test Reconstruction Error (Frobenius Norm)')
plt.ylabel('n_components')
plt.xlabel('Initialization Method')
plt.show()

# Prepare data for line plots
train_errors_df = train_errors_matrix.reset_index().melt(id_vars='index', var_name='init_method', value_name='train_error')
test_errors_df = test_errors_matrix.reset_index().melt(id_vars='index', var_name='init_method', value_name='test_error')

# Merge training and test errors for plotting
errors_df = pd.merge(train_errors_df, test_errors_df, on=['index', 'init_method'])
errors_df.rename(columns={'index': 'n_components'}, inplace=True)

# Plot Line Plot for Training and Test Errors
plt.figure(figsize=(12, 8))

# Plot for Training Errors
sns.lineplot(data=errors_df, x='n_components', y='train_error', hue='init_method', marker='o', linestyle='-', label='Train Error')

# Plot for Test Errors
sns.lineplot(data=errors_df, x='n_components', y='test_error', hue='init_method', marker='s', linestyle='--', label='Test Error')

plt.title('Reconstruction Error (Frobenius Norm) vs. n_components for Each Initialization Method')
plt.xlabel('Number of Components (n_components)')
plt.ylabel('Frobenius Norm')
plt.legend(title='Initialization Method')
plt.grid(True)
plt.show()

# Unpack the best combination
best_n_components, best_init_method = best_combination
best_result = results[best_combination]

H_best = best_result['H']
X_test_reconstructed_best = best_result['X_test_reconstructed']

# Function to display a grid of images
def plot_images(images, titles, h, w, n_row=4, n_col=4, suptitle="Images"):
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

# Function to display original and reconstructed images side by side
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

# Plot the basis images for the best combination
titles_best = [f"Comp {i+1}" for i in range(best_n_components)]
plot_images(
    H_best, 
    titles_best, 
    h=64, 
    w=64, 
    n_row=int(np.ceil(best_n_components / 8)), 
    n_col=8,  # Adjust based on n_components
    suptitle=f"NMF Basis Images (n_components={best_n_components}, init='{best_init_method}')"
)

# Plot original vs reconstructed test images for the best combination
plot_reconstructed(
    X_test, 
    X_test_reconstructed_best, 
    n_images=10, 
    suptitle=f"Original vs Reconstructed Test Images (n_components={best_n_components}, init='{best_init_method}')"
)

# Optional: Additional Scatter Plot for Comprehensive Visualization
# Each point represents a combination, colored by n_components
combination_list = list(results.keys())
train_errors_list = [results[comb]['train_error'] for comb in combination_list]
test_errors_list = [results[comb]['test_error'] for comb in combination_list]
n_components_list = [comb[0] for comb in combination_list]
init_methods_list = [comb[1] for comb in combination_list]

plt.figure(figsize=(12,8))
scatter = plt.scatter(train_errors_list, test_errors_list, 
                      c=n_components_list, cmap='viridis', alpha=0.7, edgecolors='w', s=100)

for i, (n_comp, init) in enumerate(combination_list):
    plt.annotate(f"{n_comp}-{init}", (train_errors_list[i], test_errors_list[i]), fontsize=8)

plt.xlabel('Training Reconstruction Error (Frobenius Norm)')
plt.ylabel('Test Reconstruction Error (Frobenius Norm)')
plt.title('Reconstruction Errors Across n_components and Init Methods')
plt.colorbar(scatter, label='n_components')
plt.grid(True)
plt.show()



