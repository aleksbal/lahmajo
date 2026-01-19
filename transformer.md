# Transformer forward pass (single token, collapsed, with shapes)

We compute the next-token distribution for **one token at position `t`**, using KV cache.

---

## 1️⃣ Multi-Head Self-Attention (per head)

Let:
- Model dimension: `d`
- Number of heads: `H`
- Head dimension: `d_h = d / H`
- Cache length (past tokens): `N`

For each head `h = 1 … H`:

### Inputs
- `Q_t^(h) ∈ ℝ^{1 × d_h}`  
- `K_cache^(h) ∈ ℝ^{N × d_h}`
- `V_cache^(h) ∈ ℝ^{N × d_h}`

### Attention computation
\[
\boxed{
a_t^{(h)}
=
\underbrace{\text{softmax}\!\left(
\frac{
Q_t^{(h)} \; K_{\text{cache}}^{(h)\top}
}{
\sqrt{d_h}
}
\right)}_{\;∈\;ℝ^{1×N}}
\;
V_{\text{cache}}^{(h)}
}
\]

### Output
\[
a_t^{(h)} ∈ ℝ^{1 × d_h}
\]

---

## 2️⃣ Merge heads (concatenate + output projection)

### Concatenation
\[
A_t
=
\text{concat}\!\left(
a_t^{(1)}, \dots, a_t^{(H)}
\right)
∈ ℝ^{1 × d}
\]

### Output projection
\[
y_t = A_t W_O
\quad,\quad
W_O ∈ ℝ^{d × d}
\]

\[
y_t ∈ ℝ^{1 × d}
\]

---

## 3️⃣ Residual connection + LayerNorm

\[
z_t = \text{LN}(x_t + y_t)
\quad∈\;ℝ^{1 × d}
\]

---

## 4️⃣ MLP / Feed-Forward Network (per token)

### Definition
\[
\text{MLP}(z_t)
=
W_2 \, \sigma(W_1 z_t + b_1) + b_2
\]

### Shapes
- `W₁ ∈ ℝ^{d × 4d}`
- `W₂ ∈ ℝ^{4d × d}`
- `σ` = GELU / ReLU

### Output
\[
\text{MLP}(z_t) ∈ ℝ^{1 × d}
\]

---

## 5️⃣ Residual + LayerNorm (FFN output)

\[
u_t
=
\text{LN}\!\left(
z_t + \text{MLP}(z_t)
\right)
\quad∈\;ℝ^{1 × d}
\]

➡ **`u_t` is the output of one transformer layer**

---

## 6️⃣ Stack layers

Repeat steps **1–5** for `L` layers:

\[
u_t^{(L)} ∈ ℝ^{1 × d}
\]

---

## 7️⃣ Vocabulary projection → logits → probabilities

### Logits
\[
\ell_t
=
u_t^{(L)} W_{\text{vocab}}
\quad,\quad
W_{\text{vocab}} ∈ ℝ^{d × |V|}
\]

\[
\ell_t ∈ ℝ^{1 × |V|}
\]

### Probabilities
\[
p_t = \text{softmax}(\ell_t)
\quad∈\;ℝ^{1 × |V|}
\]

---

## ✅ Mental checklist (shapes you should “see”)

- Attention scores: `1 × N`
- Attention output (per head): `1 × d_h`
- After merge: `1 × d`
- MLP input/output: `1 × d`
- Logits: `1 × |V|`
- Probabilities: `1 × |V|`