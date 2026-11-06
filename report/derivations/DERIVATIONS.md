---
title: "Nano derivations"
subtitle: "Every backward rule in nano/, derived before it was coded"
---

Each section is tagged `[Dx.y]`; the same tag appears in the docstring of the op that
implements it, so the code and the math can be audited side by side.

> **Handwritten scans.** The PRD asks for each derivation to be done on paper first and the
> scan kept in the repo. Scans go in `report/derivations/scans/` named by tag (e.g.
> `D8.png`). This file is the typeset version they link to.

Notation. Upper-case bold letters are arrays. For a scalar loss $L$ and any array
$\mathbf X$, $\bar{\mathbf X} = \partial L / \partial \mathbf X$ is the *adjoint* of
$\mathbf X$, an array of the same shape. An op $\mathbf Z = f(\mathbf X_1,\dots,\mathbf X_k)$ is
differentiable in Nano once we know its **vector-Jacobian product** (VJP)
$\bar{\mathbf Z} \mapsto (\bar{\mathbf X}_1, \dots, \bar{\mathbf X}_k)$ with
$\bar{\mathbf X}_i = J_i^\top \bar{\mathbf Z}$. We never form $J_i$.

---

## D1. Reverse mode on a DAG

The forward pass builds a directed acyclic graph whose nodes are op outputs. By the
multivariate chain rule, for a node $\mathbf X$ with consumers $\mathbf Z_1,\dots,\mathbf Z_m$

$$\bar{\mathbf X} = \sum_{j=1}^{m} J_{\mathbf Z_j,\mathbf X}^\top \, \bar{\mathbf Z}_j .$$

Two consequences fix the algorithm in `Tensor.backward`:

1. **Order.** $\bar{\mathbf X}$ is complete only after every consumer's adjoint is
   complete. A reverse topological order (children before parents) guarantees this; we
   compute it with an iterative DFS so a 5,000-op chain does not hit Python's recursion
   limit (`tests/test_tensor.py::test_deep_chain_has_no_recursion_limit`).
2. **Accumulation.** Contributions from different consumers are **summed**, which is why
   `y = x * x + x` gives $\bar x = 2x + 1$ and why gradients accumulate across `backward`
   calls (PyTorch semantics).

Cost: each node's VJP runs once, so a backward pass costs a constant multiple of the
forward pass, independent of the number of parameters.

## D2. Broadcasting and `unbroadcast`

NumPy broadcasting maps $\mathbf x$ of shape $s$ to $B(\mathbf x)$ of shape $S$ by
(a) prepending axes and (b) repeating along axes where $s_i = 1$. $B$ is linear, so its
VJP is its adjoint $B^\top$, defined by $\langle B\mathbf x, \mathbf g\rangle = \langle \mathbf x, B^\top \mathbf g\rangle$.
Each element $x_k$ appears in $B(\mathbf x)$ once per copy, hence

$$B^\top(\mathbf g) = \text{sum of } \mathbf g \text{ over the prepended axes and over every axis with } s_i=1 \text{ (keepdims).}$$

That is all `unbroadcast(grad, shape)` does, and it is applied to **every** input gradient
of **every** op in one place (`Tensor.backward`). Ops therefore return full-shape gradients
and never handle broadcasting themselves. Verified by the adjoint identity on random
shape pairs (`tests/test_properties.py::test_unbroadcast_is_adjoint_of_broadcast`).

## D3. Elementwise arithmetic and reductions

For elementwise $z = f(a, b)$ the Jacobian is diagonal, so the VJP is an elementwise product.

| Tag | Op | Forward | $\bar a$ | $\bar b$ |
|---|---|---|---|---|
| D3.1 | add / sub / neg | $a \pm b$, $-a$ | $\bar z$ | $\pm\bar z$ |
| D3.2 | mul | $ab$ | $\bar z\, b$ | $\bar z\, a$ |
| D3.3 | div | $a/b$ | $\bar z / b$ | $-\bar z\, a / b^2$ |
| D3.4 | pow (constant $p$) | $a^p$ | $\bar z\, p\, a^{p-1}$ | — |
| D3.5 | exp / log | $e^a$, $\ln a$ | $\bar z\, e^a$ (reuses the output), $\bar z / a$ | — |

**D3.6 Reductions.** $z = \sum_{i\in A} a_i$ over axes $A$: $\partial z/\partial a_i = 1$, so
$\bar a = \text{broadcast}(\bar z)$ after re-inserting the reduced axes (the dual of D2:
reduction and broadcasting are adjoint). `mean` divides by $n = \prod_{i\in A} \text{shape}_i$.

## D4. Activations

**D4.1 ReLU.** $z = \max(a,0)$, $\bar a = \bar z\,\mathbb 1[a>0]$. At $a = 0$ ReLU is not
differentiable; we use the subgradient $0$ (as PyTorch does). Central differences at the
kink return $1/2$, which is the documented FD blind spot (failure gallery F5).

**D4.2 tanh.** $\frac{d}{da}\tanh a = 1 - \tanh^2 a = 1 - z^2$, so the VJP reuses the output.
Bug B4 in the bug hunt replaces $1 - z^2$ with $1 - z$.

**D4.3 sigmoid.** $\sigma(a) = 1/(1+e^{-a})$ is computed as $\exp(-\operatorname{softplus}(-a))$
with $\operatorname{softplus}(x) = \log(1+e^x)$ evaluated by `logaddexp`, which never
overflows. $\sigma'(a) = \sigma(a)(1-\sigma(a))$, at most $1/4$: the origin of vanishing
gradients in deep sigmoid nets (D12).

## D5. Linear algebra and shape ops

**D5.1 matmul.** $\mathbf Z = \mathbf A\mathbf B$, $Z_{ij} = \sum_k A_{ik}B_{kj}$. Then
$\bar A_{ik} = \sum_j \bar Z_{ij} B_{kj}$ and $\bar B_{kj} = \sum_i A_{ik}\bar Z_{ij}$:

$$\bar{\mathbf A} = \bar{\mathbf Z}\mathbf B^\top, \qquad \bar{\mathbf B} = \mathbf A^\top \bar{\mathbf Z}.$$

Batched operands transpose only the last two axes (`swapaxes(-1,-2)`, not `.T`); shared
batch axes are summed by `unbroadcast`. `Linear` computes $\mathbf X\mathbf W^\top + \mathbf b$ with
$\mathbf W$ in PyTorch's (out, in) layout, so it is built from transpose, matmul and add.

**D5.2 reshape** relabels elements without changing them: $\bar{\mathbf a} = \text{reshape}(\bar{\mathbf z}, \text{shape}(\mathbf a))$.

**D5.3 transpose** with permutation $\pi$: $Z_{i_{\pi(1)}\dots} = A_{i_1\dots}$, so the VJP
applies $\pi^{-1}$ (`argsort(axes)`). Using `reshape` here is shape-correct and wrong:
bug B2.

**D5.4 getitem** gathers $z_j = a_{\iota(j)}$. Its adjoint scatters:
$\bar a_k = \sum_{j:\iota(j)=k} \bar z_j$. With repeated indices the sum matters, so the
VJP uses `np.add.at` (unbuffered), not fancy-index assignment.

## D6. max and logsumexp

**D6.1 max.** Away from ties, $z = a_{k^*}$ and $\bar a = \bar z\, e_{k^*}$. With $m$ tied
maxima the function is not differentiable; we split the gradient equally,
$\bar a_k = \bar z\,\mathbb 1[a_k = z]/m$, matching `torch.amax`.

**D6.2 logsumexp.** $z = \log\sum_i e^{a_i} = m + \log\sum_i e^{a_i - m}$ for any $m$; taking
$m=\max_i a_i$ keeps every exponent $\le 0$ so nothing overflows. $\partial z/\partial a_i =
e^{a_i}/\sum_j e^{a_j} = \operatorname{softmax}(a)_i$.

## D7. Cross entropy

**D7.1 Fused softmax cross entropy.** For logits $\mathbf z\in\mathbb R^{N\times C}$ and labels $y_n$,

$$L = \frac1N\sum_n \Big(\operatorname{logsumexp}(\mathbf z_n) - z_{n,y_n}\Big),\qquad
\frac{\partial L}{\partial z_{nc}} = \frac1N\big(p_{nc} - \mathbb 1[c = y_n]\big),$$

with $p_n = \operatorname{softmax}(\mathbf z_n)$ from D6.2. The backward is one subtraction
instead of chaining softmax, log and gather, and it is exact even when some $p_{nc}$
underflow to $0$. Bug B5 drops the $1/N$.

**D7.2 Why the composed version overflows.** $\operatorname{softmax}\to\log\to\text{NLL}$
without the max shift evaluates $e^{z}$ directly. $e^{z}$ overflows to `inf` once
$z > \ln(\text{max float})$: $88.7$ in float32 and $709.8$ in float64;
the sum of exponentials overflows a little earlier. Then $\infty/\infty = $ NaN.
Measured in the failure gallery (F2).

## D8. Convolution

**D8.1 im2col.** For input $\mathbf X\in\mathbb R^{N\times C\times H\times W}$, kernel $k_h\times k_w$,
stride $s$ and padding $p$, the output size is $O_h = \lfloor (H+2p-k_h)/s\rfloor + 1$ (same for
width). im2col builds $\mathbf U\in\mathbb R^{N\times Ck_hk_w\times O_hO_w}$ whose column
$(o_h,o_w)$ is the flattened receptive field at that output pixel. It is a linear map
$\mathbf U = M\mathbf X$ that copies (with repetition when windows overlap).

**D8.2 col2im is the adjoint of im2col.** Because $M$ only copies, $M^\top$ adds every column
entry back to the pixel it came from: overlapping windows **accumulate**. The identity
$\langle M\mathbf X, \mathbf C\rangle = \langle \mathbf X, M^\top\mathbf C\rangle$ is tested directly
(`tests/test_conv.py::test_col2im_is_adjoint_of_im2col`). Bug B3 shifts the scatter by one
pixel.

**D8.3 conv2d.** With $\mathbf W_m\in\mathbb R^{O\times Ck_hk_w}$ the reshaped kernel,
$\mathbf Y_n = \mathbf W_m \mathbf U_n + \mathbf b\mathbf 1^\top$. By D5.1 and D3.6:

$$\bar{\mathbf W}_m = \sum_n \bar{\mathbf Y}_n \mathbf U_n^\top,\qquad
\bar{\mathbf U}_n = \mathbf W_m^\top \bar{\mathbf Y}_n,\qquad
\bar{\mathbf X} = M^\top \bar{\mathbf U},\qquad
\bar{\mathbf b} = \sum_{n,o_h,o_w}\bar{\mathbf Y}.$$

The naive-loop oracle `conv2d_naive` implements the definition
$Y_{n,o,i,j} = b_o + \sum_{c,u,v} W_{o,c,u,v} X^{pad}_{n,c,is+u,js+v}$ and its derivative
term by term with explicit loops; it shares no code with the im2col path, which makes it
an independent check (A6, difference $< 10^{-12}$).

**D8.4 max-pool** is a max (D6.1) over each window, except that the gradient goes to the
**first** argmax only (PyTorch's choice). The routing is a scatter into window columns
followed by col2im, so overlapping windows ($s < k$) accumulate correctly.

## D9. Initialisation

**D9.1 Variance recursion.** For a layer $h^{l} = \phi(\mathbf W^{l} h^{l-1})$ with i.i.d.
zero-mean weights of variance $\sigma_l^2$ and fan-in $n_{l-1}$,
$\operatorname{Var}(\mathbf W^l h^{l-1})_i = n_{l-1}\sigma_l^2\, \mathbb E[(h^{l-1})^2]$.
Backward, the same argument with fan-out gives
$\operatorname{Var}(\bar h^{l-1}) = n_l\,\sigma_l^2\,\mathbb E[\phi'^2]\operatorname{Var}(\bar h^{l})$.
Signals neither vanish nor explode only if the per-layer gain is 1.

**D9.2 Xavier (Glorot).** For $\phi$ linear near 0 (tanh), the forward and backward
conditions are $n_{l-1}\sigma^2 = 1$ and $n_l\sigma^2 = 1$; the compromise is
$\sigma^2 = 2/(n_{\text{in}}+n_{\text{out}})$.

**D9.3 He (Kaiming).** ReLU zeroes half its inputs, so $\mathbb E[h^2] = \tfrac12\operatorname{Var}(\text{pre-activation})$
and the forward condition becomes $\tfrac12 n_{\text{in}}\sigma^2 = 1$, i.e. $\sigma^2 = 2/n_{\text{in}}$.

**Small normal** ($\sigma = 0.01$, width 128) has gain $128\cdot 10^{-4}\cdot\mathbb E[\phi'^2] \ll 1$:
signals shrink geometrically with depth. This is the cell E2 expects to fail first.

## D10. Optimizers and the stability bound

**D10.1 SGD:** $w \leftarrow w - \eta g$ (with L2 weight decay, $g \leftarrow g + \lambda w$).

**D10.2 Momentum and Nesterov** (PyTorch form): $v \leftarrow \mu v + (1-\tau) g$ (first step
$v = g$); heavy ball uses $w \leftarrow w - \eta v$; Nesterov uses the look-ahead
$w \leftarrow w - \eta(g + \mu v)$.

**D10.3 Adam:** $m \leftarrow \beta_1 m + (1-\beta_1)g$, $v \leftarrow \beta_2 v + (1-\beta_2)g^2$,
bias corrections $\hat m = m/(1-\beta_1^t)$, $\hat v = v/(1-\beta_2^t)$, then
$w \leftarrow w - \eta\,\hat m/(\sqrt{\hat v}+\epsilon)$ (the $\epsilon$ is added to $\sqrt{v}/\sqrt{1-\beta_2^t}$
as in PyTorch, which matters at the $10^{-12}$ level).

**D10.4 AdamW** decouples weight decay: $w \leftarrow w(1-\eta\lambda)$, then the Adam step
without the L2 term. With adaptive denominators, coupled L2 decays large-gradient weights
less; decoupling restores uniform decay.

**D10.5 The quadratic bound (failure gallery F4).** For $L(w)=\tfrac12(w-w^*)^\top H (w-w^*)$,
gradient descent gives $w_{t+1}-w^* = (I-\eta H)(w_t-w^*)$. Along the eigenvector of
$\lambda_{\max}$ the error is multiplied by $1-\eta\lambda_{\max}$ each step, so GD converges
iff $|1-\eta\lambda| < 1$ for all eigenvalues, i.e.

$$\eta < \frac{2}{\lambda_{\max}(H)}.$$

For least squares $L = \tfrac1{2N}\lVert Xw-y\rVert^2$, $H = X^\top X/N$. F4 measures the
empirical edge by bisection and compares it with $2/\lambda_{\max}$.

## D11. Finite-difference error (E6)

Taylor expansion gives the central difference
$\frac{f(x+h)-f(x-h)}{2h} = f'(x) + \frac{h^2}{6}f'''(\xi)$ (truncation, $O(h^2)$). In floating
point each evaluation carries rounding error about $\epsilon |f|$, which the division by
$2h$ amplifies to $\epsilon|f|/h$. The total error

$$E(h) \approx \frac{h^2}{6}|f'''| + \frac{\epsilon |f|}{h}$$

is minimised at $h^* = (3\epsilon|f|/|f'''|)^{1/3} \sim \epsilon^{1/3}$: about $6\times10^{-6}$ in
float64 ($\epsilon = 2.2\times10^{-16}$) and $5\times10^{-3}$ in float32 ($\epsilon = 1.2\times10^{-7}$),
with minimum error $\sim\epsilon^{2/3}$: $\approx 4\times10^{-11}$ and $\approx 2\times10^{-5}$. This is
the U-curve measured in E6. It also explains two observations there: linear ops
(matmul, conv2d) have $f'''=0$, so only rounding remains and the best $h$ is as large as
the grid allows; and float32 cannot certify gradients to better than about $10^{-5}$, which
is why the precision contract is stated in float64.

## D12. Why depth breaks gradient flow (theory behind E2)

For an MLP $h^l = \phi(a^l)$, $a^l = W^l h^{l-1}$, the backward recursion is
$\bar h^{l-1} = W^{l\top}(\phi'(a^l)\odot \bar h^l)$, so

$$\bar h^{0} = \Big(\prod_{l=1}^{D} W^{l\top}\operatorname{diag}\phi'(a^l)\Big)\,\bar h^{D},
\qquad \lVert\bar W^l\rVert = \lVert \bar a^l\rVert\,\lVert h^{l-1}\rVert .$$

The per-layer factor has typical gain $g = \sqrt{n\sigma^2\,\mathbb E[\phi'^2]}$. Over $D$
layers the gradient at the input scales like $g^D$:

* sigmoid: $\phi' \le 1/4$, so even with Xavier $g < 1$ and $\lVert\bar W^1\rVert$ decays
  exponentially in $D$;
* any activation with small-normal init: $n\sigma^2 = 0.0128$, so both activations and
  gradients collapse; the forward signal dies too, which also shrinks
  $\lVert h^{l-1}\rVert$;
* ReLU with He, tanh with Xavier: $g\approx 1$ by construction (D9), so depth costs little
  at initialisation.

E2's predictor $r = \log_{10}(\lVert\bar W^1\rVert/\lVert\bar W^{D+1}\rVert)$ is therefore
approximately $D\log_{10} g$ plus a depth-independent offset: a linear-in-depth measure of
how far the network is from dynamical isometry *at step 0*. H1 asks whether this number,
which costs four backward passes, predicts the outcome of a full training run.
