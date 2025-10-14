# GNN HIV Drug Response Prediction

## Project Title & Description

This repository contains a Python implementation of a Graph Neural Network (GNN) model for predicting HIV drug response. The project leverages PyTorch Geometric and aims to provide an effective, reproducible solution for drug response prediction using graph-based deep learning.

## Key Features & Benefits

*   **Graph Neural Network Architecture:** Utilizes state-of-the-art GNN models for capturing complex relationships in molecular data.
*   **HIV Drug Response Prediction:** Specifically designed to predict the efficacy of drugs against HIV strains.
*   **PyTorch Geometric Implementation:** Built using PyTorch Geometric, a powerful framework for graph-based deep learning.
*   **Customizable:** Offers flexibility to modify model architectures, hyperparameters, and dataset paths.
*   **Data Handling:** Implements efficient data loading and preprocessing for molecular graphs.
*   **Evaluation Metrics:** Provides tools for model evaluation using relevant drug response metrics.

## Prerequisites & Dependencies

Before you begin, ensure you have the following installed:

*   **Python:** (>=3.6)
*   **PyTorch:** (>=1.8) `pip install torch torchvision torchaudio`
*   **PyTorch Geometric:** Installation instructions at https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
*   **NumPy:** `pip install numpy`
*   **Scikit-learn:** `pip install scikit-learn`
*   **Tqdm:** `pip install tqdm`
*   **RDKit:** For molecular graph processing (optional, depending on dataset)

## Installation & Setup Instructions

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/jyotipokhrel22/gnn-hiv.git
    cd gnn-hiv
    ```

2.  **Create a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

3.  **Install required packages:**

    ```bash
    pip install -r requirements.txt
    ```
    (Create `requirements.txt` based on the packages used)

4.  **Prepare your dataset:** Place your HIV molecular and drug response data in the appropriate directory as expected by the data loader.

## Usage Examples & API Documentation (if applicable)

### Training the Model

1.  **Modify `train.py`:** Adjust dataset paths, model parameters, batch size, learning rate, and epochs as needed.

2.  **Run the training script:**

    ```bash
    python train.py
    ```

### Model Architecture

The `model.py` file defines the GNN architecture, which typically includes:

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)  # graph-level readout (example)
        x = self.lin(x)
        return x
