# Introduction

Deep neural network architectures have undergone rapid evolution since the introduction of ResNets (He et al., 2016a). As illustrated in Fig. 1(a), the structure of a single layer can be formulated as follows:

$$
x_{\ell+1} = x_{\ell} + \mathcal{F}(x_{\ell}, W_{\ell})
$$
(1)

where $x_{\ell}$ and $x_{\ell+1}$ denote the $C$-dimensional input and output of the $\ell$-th layer, respectively, and $\mathcal{F}$ represents the residual function. Although the residual function $\mathcal{F}$ has evolved over the past decade to include various operations such as convolution, attention mechanisms, and feed-forward networks, the paradigm of the residual connection has maintained its original form. Accompanying the progression of the Transformer (Vaswani et al., 2017) architecture, this paradigm has currently established itself as a fundamental design element in large language models (LLMs) (Brown et al., 2020; Liu et al., 2024b; Touvron et al., 2023).

This success is primarily attributed to the concise form of the residual connection. More importantly, early research (He et al., 2016b) revealed that the identity mapping property of the residual connection maintains stability and efficiency during large-scale training. By recursively extending the residual connection across multiple layers, Eq. (1) yields:

$$
x_{L} = x_{\ell} + \sum_{i=\ell}^{L-1} \mathcal{F}(x_i, W_i)
$$
(2)

where $L$ and $\ell$ correspond to deeper and shallower layers, respectively. The term identity mapping refers to the component $x_{\ell}$ itself, which emphasizes the property that the signal from the shallower layer maps directly to the deeper layer without any modification.

Recently, studies exemplified by Hyper-Connections (HC) (Zhu et al., 2024) have introduced a new dimension to the residual connection and empirically demonstrated its performance potential. The single-layer architecture of HC is illustrated in Fig. 1(b). By expanding the width of the residual stream and enhancing connection complexity, HC significantly increases topological complexity without altering the computational overhead of individual units with respect to FLOPs. Formally, single-layer propagation in HC is defined as:

$$
x_{\ell+1} = H^{\mathrm{res}}_{\ell} x_{\ell} + H^{\mathrm{post}\top}_{\ell} \; \mathcal{F}\bigl(H^{\mathrm{pre}}_{\ell} x_{\ell}, W_{\ell}\bigr)
$$
(3)

where $x_{\ell}$ and $x_{\ell+1}$ denote the input and output of the $\ell$-th layer, respectively. Unlike the formulation in Eq. (1), the feature dimension of $x_{\ell}$ and $x_{\ell+1}$ is expanded from $C$ to $n\times C$, where $n$ is the expansion rate. The term $H^{\mathrm{res}}_{\ell}\in\mathbb{R}^{n\times n}$ represents a learnable mapping that mixes features within the residual stream. Also as a learnable mapping, $H^{\mathrm{pre}}_{\ell}\in\mathbb{R}^{1\times n}$ aggregates features from the $nC$-dim stream into a $C$-dim layer input, and conversely, $H^{\mathrm{post}}_{\ell}\in\mathbb{R}^{1\times n}$ maps the layer output back onto the stream.

However, as the training scale increases, HC introduces potential risks of instability. The primary concern is that the unconstrained nature of HC compromises the identity mapping property when the architecture extends across multiple layers. In architectures comprising multiple parallel streams, an ideal identity mapping serves as a conservation mechanism: it ensures that the average signal intensity across streams remains invariant during both forward and backward propagation. Recursively extending HC to multiple layers via Eq. (3) yields:

$$
x_{L} = \left( \prod_{i=1}^{L-\ell} H^{\mathrm{res}}_{L-i} \right) x_{\ell} + \sum_{i=\ell}^{L-1} \left( \prod_{j=1}^{L-1-i} H^{\mathrm{res}}_{L-j} \right) H^{\mathrm{post}\top}_{i} \; \mathcal{F}\bigl(H^{\mathrm{pre}}_{i} x_{i}, W_{i}\bigr)
$$
(4)

(Equation layout adapted for Markdown/KaTeX rendering.)

---

References mentioned in this excerpt: He et al. (2016a, 2016b); Vaswani et al. (2017); Brown et al. (2020); Liu et al. (2024a, 2024b); Touvron et al. (2023); Zhu et al. (2024).
