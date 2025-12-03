# Transformer Assignment – Scaled Dot-Product Attention & Encoder Block

This repository contains solutions to two core transformer architecture components:

1. **Q1:** Scaled Dot-Product Attention (NumPy)  
2. **Q2:** Simple Transformer Encoder Block (PyTorch)

All implementations are fully functional, clean, and easy to test inside the provided Jupyter notebook.

---

## 📌 Contents
- `Attention_Transformer_Assignment.ipynb` – Main notebook with both solutions
- (Optional) `requirements.txt` – Dependencies for NumPy & PyTorch

---

## 🚀 Q1 – Scaled Dot-Product Attention (NumPy)

Implements the formula:

\[
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

### Features:
- Uses NumPy for matrix operations  
- Includes custom softmax implementation  
- Returns:
  - **Attention weights**
  - **Context vector**

### Test Example Included:
- Works with sample Q, K, V matrices  
- Prints weights + context

---

## 🚀 Q2 – Transformer Encoder Block (PyTorch)

Implements a simplified transformer encoder with:

### Components:
- **Multi-head self-attention**
- **Feed-forward network (Linear → ReLU → Linear)**
- **Add & Norm layers (LayerNorm + residual connections)**

### Dimensions:
- `d_model = 128`
- `num_heads = 8`
- `dim_ff = 512`

### Output Verification:
For a batch of **32 sentences**, each with **10 tokens**:


