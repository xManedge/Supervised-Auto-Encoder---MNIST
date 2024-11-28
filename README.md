# MNIST Supervised Autoencoders  

## Objective  
This project implements a **Supervised Autoencoder (SAE)** to classify MNIST digits. The SAE combines the benefits of unsupervised feature learning and supervised classification into a single architecture. The autoencoder is used for dimensionality reduction, while the supervised component maps encoded features to class labels.  

## Dataset Description  
The **MNIST Dataset** is used in this project. It consists of grayscale handwritten digit images (0-9), commonly used for benchmarking machine learning models.  

- **Training set:** 60,000 images.  
- **Test set:** 10,000 images.  
- **Image size:** 28x28 pixels, reshaped into a flat vector of 784 dimensions.  
- **Labels:** One-hot encoded for multi-class classification.  

### Data Preprocessing  
- Images are normalized to the range `[0, 1]`.  
- Labels are converted to one-hot encoded vectors for compatibility with the classification loss.  
- Data is batched using PyTorch's `DataLoader` for efficient training.  

## Model Architecture  

The **Supervised Autoencoder (SAE)** consists of the following components:  

1. **Encoder:**  
   - Reduces the 784-dimensional input to a low-dimensional latent space.  
   - Includes linear layers with non-linear activations (ReLU).  

2. **Decoder:**  
   - Reconstructs the original input from the latent representation.  
   - Ensures the encoder learns meaningful features for reconstruction.  

3. **Classifier:**  
   - A fully connected layer that maps the latent features to class probabilities.  
   - Uses a softmax activation for multi-class classification.  

### Hyperparameters  
- **Learning Rate:** 0.001 (tunable).  
- **Batch Size:** 32 or 128 (used for different stages of training).  
- **Loss Functions:**  
  - **Reconstruction Loss:** Mean Squared Error (MSE).  
  - **Classification Loss:** Cross-Entropy Loss.  
- **Optimizer:** Adam optimizer for efficient gradient descent.  
- **Epochs:** 40.  

## Training and Inference  

### Training  
- The training process optimizes two objectives:  
  1. Minimize reconstruction loss for the autoencoder.  
  2. Minimize classification loss for the supervised component.  
- Loss values are tracked across epochs for both training and validation.  

### Inference  
- The encoder extracts low-dimensional features from the test data.  
- The classifier predicts the class of each digit.  

**Example Output:**  
- Encoded feature shape: `(10000, <latent_dim>)`, where `<latent_dim>` is the size of the latent space.  
- Predicted class probabilities for test samples.  

## Results  
The model achieves the following results on the MNIST dataset:  

- **Reconstruction Loss:** 1.523 mse   
- **Classification Accuracy:** 94.23%  
- **Latent Feature Visualization:** The latent features are normalized and visualized to observe separability between classes.  

## Repository Contents  

- **`MNIST Supervised Autoencoders.ipynb`:**  
  A Jupyter Notebook containing the entire pipeline:  
  - Data preprocessing.  
  - Model architecture definition.  
  - Training and evaluation.  
  - Visualization of results.  

## Requirements  

### Prerequisites  
Ensure the following libraries are installed:  
- Python 3.x  
- PyTorch  
- Torchmetrics (for accuracy computation).  
- NumPy  
- Matplotlib  
- Torchvision  

Install dependencies with:  
```bash  
pip install torch torchvision torchmetrics numpy matplotlib  
```  

### Running the Code  
1. Clone this repository:  
   ```bash  
   git clone https://github.com/yourusername/mnist-supervised-autoencoders.git  
   ```  
2. Navigate to the project directory and open the Jupyter Notebook:  
   ```bash  
   cd mnist-supervised-autoencoders  
   jupyter notebook  
   ```  
3. Follow the steps in the notebook to train the model and evaluate its performance.  

## Future Work  
- Experiment with deeper architectures or different loss functions.  
- Apply the supervised autoencoder approach to other datasets (e.g., CIFAR-10).  
- Use techniques like transfer learning or pretraining for performance improvement.  
- Visualize encoded features using t-SNE or PCA for better interpretability.  

## Contributing  
Contributions are welcome! Submit a pull request or open an issue for suggestions or improvements.  
