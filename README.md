# nanoGPT

A **minimal, fast, and readable implementation of GPT-style Transformer language models**, similar to nanoGPT created by **Andrej Karpathy**.

This is the decoder-only model, which generates text in a similar style to the input file.

---

## 🧠 Model Architecture

nanoGPT implements a **decoder-only Transformer** similar to GPT-2:

* Token Embeddings + Positional Embeddings
* Multi-Head Self Attention
* Feed Forward Network (MLP)
* Layer Normalization
* Residual Connections

Mathematically, it models:

> **P(xₜ | x₁, x₂, ..., xₜ₋₁)**

using causal self-attention.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/cherry-mittal/GPT-from-scratch.git
cd nanoGPT
```

---

## 📊 Dataset Preparation

### Example: Shakespeare (Character-level)

```bash
python data/shakespeare_char/prepare.py
```

---

## 🏋️ Training

Key training features:

* Gradient accumulation
* Mixed precision (fp16 / bf16)
* Checkpointing
* Learning rate scheduling

---

## ✨ Text Generation

Generate text using a trained model:

```bash
python sample.py --out_dir=out-shakespeare-char
```

You can control:

* Temperature
* Top-k sampling
* Max tokens

---

## ⚙️ Configuration System

nanoGPT uses **Python-based configs** for full flexibility:

```python
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
batch_size = 64
learning_rate = 3e-4
```

This makes experimentation extremely fast and intuitive.

---

## 🔬 Performance

nanoGPT is optimized for speed:

* Flash Attention (when available)
* Torch compile support
* Efficient fused kernels

It can train **GPT-2 sized models** in hours instead of days on modern GPUs.

---

## 📚 Learning Resources

Highly recommended companion resources:

* 🎥 *Let's build GPT from scratch* — Andrej Karpathy (YouTube)
* 📄 *Attention Is All You Need* (Vaswani et al.)
* 📄 *GPT-2 Paper* (OpenAI)

---

## 🧪 Use Cases

* Learn how LLMs work internally
* Prototype new Transformer ideas
* Train small-to-medium GPT models
* Interview preparation for ML / LLM roles
* Research & experimentation

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. It is **not** designed as a production-ready LLM system.

---

## 🙌 Credits

Inspired by:
* Andrej Karpathy's nanoGPT
* OpenAI GPT models
* PyTorch ecosystem

---

## ⭐ Acknowledgements

If you find this repo useful, consider giving it a ⭐ and supporting open-source ML education.

Happy hacking! 🚀
