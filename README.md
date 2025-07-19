# 🛒 Node Classification in the Amazon Product Graph

This project performs **node classification** on the large-scale **Amazon product co-purchasing network** using **Graph Neural Networks (GNNs)**. Two powerful models — **GCN** and **GraphSAGE** — are implemented and compared using the [`ogbn-products`](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) dataset from the **Open Graph Benchmark (OGB)**.

---

## 🧠 What’s Inside?

- ⚡ Built using **PyTorch Geometric (PyG)**
- 🧱 Supports both **GCN** and **GraphSAGE** architectures
- 🧪 Evaluated using accuracy and F1-score on train/val/test splits
- 📊 Confusion matrix visualization for model analysis
- ✅ Fully reproducible with clear modular code

---

## 📦 Dataset Info

- **Name**: ogbn-products
- **Source**: Open Graph Benchmark
- **Description**: An Amazon product network where nodes represent products, and edges indicate that two products are frequently bought together. The task is to predict product category.

---

## 🏗️ Model Architectures

### GCN (Graph Convolutional Network)
- 2 layers of `GCNConv`
- ReLU activation between layers

### GraphSAGE
- 2 layers of `SAGEConv`
- ReLU activation between layers

---

## 📈 Evaluation Metrics

- **Accuracy**
- **F1-score (macro)**
- **Confusion Matrix**
- Results compared across:
  - Training Set
  - Validation Set
  - Test Set

---

## 🧪 How to Run

1. Clone the repository:

```bash
git clone https://github.com/Sayed-Hossein-Hosseini/amazon-node-classification.git
````

2. Install required libraries:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric ogb
pip install seaborn matplotlib scikit-learn
```

3. Run the notebook:

```bash
jupyter notebook Node_Classification_in_the_Amazon_Product_Graph.ipynb
```

## 🔮 Future Work

* Implement attention-based GNNs (e.g., GAT)
* Apply link prediction on same graph
* Use full-batch training with distributed support
* Convert model to TorchScript for deployment

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve this project.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

Created by **Sayyed Hossein Hosseini DolatAbadi**
Feel free to connect or reach out via GitHub.
