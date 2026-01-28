+++
date = '2026-01-28T16:34:21+08:00'
draft = false
title = 'Transfromer学习记录'
tags = ["transformer"]
categories = ["技术"]
math = true
ShowToc = true
+++

> 基于 llama2.c 项目的 run.c 源码注释整理

## 写在开头

一直在关注llm的相关进展，但是感觉对Transformer一知半解，前几天在ai推荐下深入阅读了 Llama 2 的纯 C 语言实现（llama2.c），主要是通过写注释+查不懂的原理的方式理解这个代码，读完感觉对transformer清晰很多，大佬还是太强了，学习到了很多之前没有注意到的细节。

本文由ai阅读我的注释后辅助撰写。

源码链接如下：https://github.com/karpathy/llama2.c

## 目录

1. [写在开头](#写在开头)
2. [Transformer 核心概念](#transformer-核心概念)
3. [模型配置与数据结构](#模型配置与数据结构)
4. [Transformer Block ](#transformer-block)
5. [位置编码：RoPE](#位置编码rope)
6. [前向传播流程](#前向传播流程)
7. [分词器：BPE](#分词器bpe)
8. [采样策略](#采样策略)
9. [推理循环](#推理循环)
10. [总结](#总结)

---

## Transformer 核心概念

### 整体架构

一个完整的大模型推理流程如下：

```
用户输入 → 分词(Tokenizer) → 词嵌入(Embedding) → 
[Block 1 → Block 2 → ... → Block N] → 
最终归一化 → 线性层投影 → Softmax 采样 → 输出 Token
```

### Decoder-Only 架构

在 Decoder-only 架构（如 GPT、Llama）中，所有的 Transformer Block 是完全线性串联的。每个 Block 的输出直接作为下一个 Block 的输入。


---

## 模型配置与数据结构

### Config：模型超参数

```c
typedef struct {
    int dim;           // 词嵌入向量的长度
    int hidden_dim;    // FFN 中间层的宽度
    int n_layers;      // Transformer Block 重复堆叠的次数
    int n_heads;       // Multi-Head Attention 中 Query 的头数
    int n_kv_heads;    // Key 和 Value 的头数（可以小于 n_heads）
    int vocab_size;    // 词表大小
    int seq_len;       // 最大序列长度（上下文窗口）
} Config;
```

**参数解释：**

- **dim**：词嵌入向量（Embedding）的长度。输入会先转化为 token，每个 token 映射为一个 dim 长度的向量。
- **hidden_dim**：在 Transformer 每一层内部，Self-Attention 后面都跟着一个全连接层（FFN）。hidden_dim 就是这个中间层的宽度。模型先将 dim 扩展到 hidden_dim（通常是 2-4 倍），进行非线性变换，再压缩回 dim。这是模型存储"知识"的主要地方。
- **n_kv_heads**：在现代模型中，KV 的头通常少于 Q 的头，实现压缩显存占用的目的，多个 Q 头共享同一个 KV 矩阵。

### TransformerWeights：模型权重

```c
typedef struct {
    float* token_embedding_table;  // (vocab_size, dim) Token 嵌入表
    float* rms_att_weight;         // (layer, dim) Attention 前的归一化权重
    float* rms_ffn_weight;         // (layer, dim) FFN 前的归一化权重
    float* wq;                     // (layer, dim, n_heads * head_size) Query 矩阵
    float* wk;                     // (layer, dim, n_kv_heads * head_size) Key 矩阵
    float* wv;                     // (layer, dim, n_kv_heads * head_size) Value 矩阵
    float* wo;                     // (layer, n_heads * head_size, dim) 输出矩阵
    float* w1;                     // (layer, hidden_dim, dim) FFN 升维矩阵（Gate）
    float* w2;                     // (layer, dim, hidden_dim) FFN 降维矩阵
    float* w3;                     // (layer, hidden_dim, dim) FFN 升维矩阵（Up）
    float* rms_final_weight;       // (dim,) 最终归一化权重
    float* wcls;                   // 输出分类器（可能与 embedding 共享）
} TransformerWeights;
```

**权重共享：**
在许多 Transformer 模型中，输入词嵌入层和输出分类层是共享权重的。如果 `wcls` 直接指向 `token_embedding_table`，可以节省大量显存。

### RunState：运行时激活值

```c
typedef struct {
    float *x;            // 当前激活值 (dim,)
    float *xb;           // 残差分支缓冲区 (dim,)
    float *xb2;          // 额外缓冲区 (dim,)
    float *hb;           // FFN 隐藏层缓冲区 (hidden_dim,)
    float *hb2;          // FFN 隐藏层缓冲区 (hidden_dim,)
    float *q;            // Query (dim,)
    float *k;            // Key (dim,)
    float *v;            // Value (dim,)
    float *att;          // 注意力分数 (n_heads, seq_len)
    float *logits;       // 输出 logits
    float* key_cache;    // KV 缓存 (layer, seq_len, dim)
    float* value_cache;  // KV 缓存 (layer, seq_len, dim)
} RunState;
```

**KV Cache 的重要性：**
模型每产生一个新词，都要回看之前所有的词。为了不重复计算前面所有词的 K 和 V，我们将它们永久存在这两个"记忆库"中。

**显存占用示例：**
如果一个模型的 `dim=4096, n_layers=32, seq_len=4096`，且不使用 GQA（即 `kv_dim=4096`）：

- 单个 Cache 的大小 = 32 × 4096 × 4096 × 4 字节 ≈ 2 GB
- 两个 Cache 加起来就需要 4 GB 显存/内存


---

## Transformer Block

Transformer Block 由两个子层组成，每个子层都辅助有**残差连接（Residual Connection）**和**层归一化（Layer Normalization）**。

### 1. Multi-Head Attention（多头自注意力层）

这是 Block 的第一个子层，负责捕捉序列中不同位置之间的依赖关系。

**简单理解：**

- 输入是"词典里的词"
- Attention 过程是"读句子的过程"
- 输出则是"放在句子里理解之后的词"

#### 计算流程

**步骤 1：线性投影**

输入的特征张量 $X \in \mathbb{R}^{seq\_len \times dim}$ 分别乘以三个权重矩阵 $W^Q, W^K, W^V$，生成查询（Query）、键（Key）和值（Value）矩阵。

**步骤 2：多头拆分**

将上面得到的三个矩阵的维度 $dim$ 拆分为 $n_{heads}$ 个子空间，每个头的维度为 $d_k = dim / n_{heads}$。

- 每个 `[Batch, Seq_Len, dim]` 变成了 `[Batch, Seq_Len, n_heads, d_k]`
- 也就是说原来的 dim 长度的向量，切成了 n_heads 个

为了并行，对维度进行置换：

- `[Batch, Seq_Len, n_heads, d_k]` → `[Batch, n_heads, Seq_Len, d_k]`
- 仅交换位置，`Batch` 和 `n_heads` 看作是批处理维度，同时在 `n_heads` 个子空间内并行计算各自的注意力

**步骤 3：注意力计算**

在每个头内部计算点积注意力，这里 $Q_i, K_i, V_i$（形状均为 `[Seq_Len, d_k]`）：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**详细分解：**

1. **$QK^T$**：`[Seq_Len, d_k] × [d_k, Seq_Len]` → `[Seq_Len, Seq_Len]`

   - 被称为**注意力得分矩阵（Attention Score Matrix）**
   - 元素 $(i, j)$：代表序列中第 $i$ 个单词与第 $j$ 个单词之间的关联强度

2. **缩放操作** $\frac{1}{\sqrt{d_k}}$：

   - 当维度 $d_k$ 非常大时，点积结果的方差会变得很大
   - 假设 $Q$ 和 $K$ 的分量都是均值为 0、方差为 1 的独立随机变量，则 $QK^T$ 的点积均值为 0，方差为 $d_k$
   - 除以 $\sqrt{d_k}$ 将方差重新缩放到 1，使梯度保持平稳

3. **归一化处理** $\text{softmax}(\cdot)$：

   - 假设输入向量为 $\mathbf{z} = [z_1, z_2, \dots, z_n]$，Softmax 运算对其中每一个元素 $z_i$ 的计算公式如下：

   $$
   \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
   $$

   - **物理意义**：将原始得分转化为概率分布，即所有的权重都变为正数且和为 1。这决定了在生成最终输出时，每个位置的信息应该贡献多少比例。

4. **加权求和** $\cdot V$：

   - 根据计算出的注意力权重，对 $V$ 向量进行线性加权
   - 模型最终会选择性地从 $V$ 中提取出与当前 $Q$ 最相关的信息

**步骤 4：多头拼接与投影**

将所有头的输出拼接回 $dim$ 维度，再通过一个线性投影矩阵 $W^O$ 进行映射。

#### 维度变化总结

假设输入序列长度为 $L$，特征维度为 $d$：

- 输入映射：$Q(L \times d_k), K(L \times d_k), V(L \times d_v)$
- 得分矩阵 ($QK^T$)：$(L \times d_k) \times (d_k \times L) = (L \times L)$
- Softmax 概率：$(L \times L)$
- 最终输出 ($\text{Prob} \times V$)：$(L \times L) \times (L \times d_v) = (L \times d_v)$


### 2. Feed-Forward Network（前馈神经网络层）

FFN 负责对每个词本身的特征进行深度加工和非线性变换。

**简单理解：**

- 输入是"初步理解了语境的词"
- FFN 过程是"对照脑海中的知识库进行深加工和逻辑验证的过程"
- 输出则是"最终定型、具备深层逻辑含义的知识表达"

#### 公式

$$
\text{FFN}(x) = \text{Activation}(xW_1 + b_1)W_2 + b_2
$$

其中：

- $x$：是该层的输入特征，形状为 `[Seq_Len, dim]`
- $W_1$：第一层线性变换的权重矩阵，形状为 `[dim, hidden_dim]` —— **升维 (Expansion)**
- $W_2$：第二层线性变换的权重矩阵，形状为 `[hidden_dim, dim]` —— **降维 (Projection)**
- $b_1, b_2$：偏置项（现代大模型如 Llama 往往会去掉偏置项以提高泛化能力）

#### 激活函数

不同模型使用不同的激活函数：

1. **ReLU**（原始 Transformer）：$\max(0, x)$

   - 简单高效，但存在神经元"坏死"的问题

2. **GELU**（BERT, GPT-3）：高斯误差线性单元

   - 在 0 附近更平滑，目前是中等规模模型的标准配置

3. **SwiGLU**（Llama, PaLM）：现代大模型的主流选择

   - 结合了 Swish 激活函数和门控线性单元（GLU）
   - 计算方式：

   $$
   \text{SwiGLU}(x) = (\text{Swish}(xW_1) \otimes xW_3)W_2
   $$

   其中 $\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$

   - 这种设计增加了额外的参数，但显著提升了模型的收敛速度和最终表现

#### 代码实现

```c
// 升维
matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

// SwiGLU 激活
for (int i = 0; i < hidden_dim; i++) {
    float val = s->hb[i];
    // SiLU(x) = x * σ(x)
    val *= (1.0f / (1.0f + expf(-val)));
    // 逐元素乘以 w3(x)
    val *= s->hb2[i];
    s->hb[i] = val;
}

// 降维
matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
```

### 3. 残差连接与层归一化

#### 残差连接（Residual Connection）

$$
\text{Output} = x + f(x)
$$

其中 $x$ 为输入，$f$ 为 Attention 或 FFN。

**物理意义**：保留原始信息，防止在交流中"迷失自我"。

#### 层归一化（Layer Normalization）

将神经元的输出调整到一个合理的分布范围（通常是均值为 0，方差为 1）。

$$
\text{LN}(x_i) = \gamma \cdot \hat{x}_i + \beta
$$

其中：

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

- $\mathbf{x}$：输入向量，在 Transformer 中其维度即为 dim（如 512 或 4096）
- $\mu$：均值
- $\sigma^2$：方差
- $\epsilon$：数值稳定性项，确保分母不为零
- $\gamma$（Gamma）：维度与 dim 一致的向量，初始化通常为全 1。模型通过训练来调整每一维特征的缩放比例
- $\beta$（Beta）：维度与 dim 一致的向量，初始化通常为全 0。模型通过训练来调整每一维特征的基准偏移

#### RMSNorm（Llama 使用）

RMSNorm（Root Mean Square Layer Normalization，均方根归一化）是 LayerNorm 的简化版本：

1. 计算均方根：
   $$
   \text{RMS}(x) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2 + \epsilon}
   $$

2. 归一化：
   $$
   \bar{x}_i = \frac{x_i}{\text{RMS}(x)}
   $$

3. 缩放：
   $$
   y_i = \bar{x}_i \cdot \gamma_i
   $$

**代码实现：**

```c
void rmsnorm(float* o, float* x, float* weight, int size) {
    // 计算平方和
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f; // epsilon
    ss = 1.0f / sqrtf(ss);
    // 归一化并缩放
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}
```


### 结构组装流水线（Pre-Norm）

一个 Transformer Block 的完整流程（以 Pre-Norm 为例）：

1. **第一层归一化（LayerNorm 1）**：对输入向量进行标准化，平滑数值
2. **多头自注意力（Attention）**：向量在这里进行"横向交流"，获取上下文
3. **第一层残差相加（Residual Add）**：将 Attention 的输出与最原始的输入直接相加
4. **第二层归一化（LayerNorm 2）**：再次标准化，为深度计算做准备
5. **前馈神经网络（FFN）**：向量在这里进行"纵向深挖"，对照知识库提取语义
6. **第二层残差相加（Residual Add）**：将 FFN 的输出与进入 FFN 前的状态相加

**数学表达：**

$$
x_1 = x + \text{Attention}(\text{LayerNorm}_1(x))
$$

$$
x_{out} = x_1 + \text{FFN}(\text{LayerNorm}_2(x_1))
$$

---

## 位置编码：RoPE

### 为什么需要位置编码？

Attention 机制本身是**位置无关**的。如果不加位置信息，模型无法区分"我爱你"和"你爱我"。

### RoPE（Rotary Positional Embedding）

RoPE 通过在向量空间中旋转 Q 和 K，使得它们的点积自动包含相对位置信息。

**核心思想**：如果这个词处于句子的第 $m$ 个位置，我们就把这个点绕原点旋转 $m \cdot \theta$ 角度。

### 数学推导

假设我们有一个二维向量 $\mathbf{x} = [x_1, x_2]$，它处于位置 $m$。我们定义一个旋转矩阵 $\mathbf{R}_m$：

$$
\mathbf{R}_m = \begin{pmatrix} 
\cos(m\theta) & -\sin(m\theta) \\ 
\sin(m\theta) & \cos(m\theta) 
\end{pmatrix}
$$

旋转后的向量为：

$$
\mathbf{x}' = \mathbf{R}_m \mathbf{x}
$$

### 为什么这能代表相对位置？

这是 RoPE 最精妙的地方。当我们计算 Query ($Q$) 在位置 $m$ 和 Key ($K$) 在位置 $n$ 的点积时：

$$
\text{Score}(m, n) = (\mathbf{R}_m \mathbf{q})^T (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^T \mathbf{R}_m^T \mathbf{R}_n \mathbf{k} = \mathbf{q}^T \mathbf{R}_{n-m} \mathbf{k}
$$

**结论**：点积的结果只与 $n-m$（即两个词的相对距离）有关。这意味着，尽管我们在每一层对 $Q$ 和 $K$ 做了绝对位置的旋转，但注意力机制捕捉到的却是相对距离。

### 多维扩展与计算优化

对于一个 $d$ 维的高维向量，RoPE 会将其拆分成 $d/2$ 个二维对。每一对都会根据预设的频率 $\theta_i$ 进行不同速度的旋转：

$$
\theta_i = 10000^{-2i/d}
$$

在实际代码实现中，我们不会真的去乘一个巨大的旋转矩阵（太慢了），而是利用复数乘法的简化形式：

$$
\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} \to \begin{pmatrix} 
x_1 \cos(m\theta) - x_2 \sin(m\theta) \\ 
x_1 \sin(m\theta) + x_2 \cos(m\theta) 
\end{pmatrix}
$$

### 代码实现

```c
// RoPE 相对位置编码：在每个头内旋转 q 和 k
for (int i = 0; i < dim; i+=2) {
    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    int rotn = i < kv_dim ? 2 : 1; // 2 = q & k, 1 = q only
    for (int v = 0; v < rotn; v++) {
        float* vec = v == 0 ? s->q : s->k;
        float v0 = vec[i];
        float v1 = vec[i+1];
        vec[i]   = v0 * fcr - v1 * fci;
        vec[i+1] = v0 * fci + v1 * fcr;
    }
}
```

**为什么使用 `head_dim` 而不是全局索引 `i`？**

一个 dim 被切割成多个头的部分，每个头对应位置的频率相同。如果直接用全局索引 i 来计算频率，那么第一个头的旋转频率会非常快，而最后一个头的旋转频率会极其慢。这会导致不同的头对"位置"的理解完全不同，模型就乱套了。


---

## 前向传播流程

### 完整的模型架构

```
输入端：词嵌入 + 位置编码
    ↓
[Block 1]
    ↓
[Block 2]
    ↓
   ...
    ↓
[Block N]
    ↓
输出端：最终归一化 + 线性层 + Softmax
```

### 输入端：词嵌入与位置编码

在进入第一个 Block 之前，数据需要经过：

1. **词嵌入层（Token Embedding）**：将 Token ID 转换成维度为 `dim` 的向量
2. **位置编码（Positional Encoding / RoPE）**：现代模型（如 Llama）通常在每一层的 Attention 内部直接应用旋转位置嵌入（RoPE），让模型知道词与词之间的相对距离

### 输出端：输出头

在最后一个 Block 输出向量后，需要将其变回人类可读的文字：

1. **最终归一化（Final LayerNorm / RMSNorm）**：对最后一层的输出进行最后的数值校准
2. **线性层（Linear/Language Model Head）**：将维度从 `dim` 映射回巨大的词表维度（`vocab_size`）。得到一个长度为 `vocab_size` 的长向量 $z$，称为 **Logits（原始得分）**。向量中的每一个数代表了对应单词的"得分"，得分越高，模型认为该词出现的可能性越大
3. **Softmax**：计算出下一个 Token 出现的概率分布

### 优化组件

- **KV Cache**（推理时存在）：在推理过程中，为了加速，模型会在内存中开辟一块空间，缓存每一层已经计算过的 Key 和 Value 向量
- **激活检测与 Dropout**：在训练阶段存在，用于防止过拟合

### 前向传播代码流程

```c
float* forward(Transformer* transformer, int token, int pos) {
    // 1. 获取 token 的嵌入向量
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));
    
    // 2. 遍历所有层
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        // 2.1 Attention 前的 RMSNorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
        
        // 2.2 计算 QKV
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
        
        // 2.3 应用 RoPE
        // ... (旋转 Q 和 K)
        
        // 2.4 多头注意力
        for (h = 0; h < p->n_heads; h++) {
            // 计算注意力分数
            for (int t = 0; t <= pos; t++) {
                score = dot_product(q, k[t]) / sqrt(head_size);
                att[t] = score;
            }
            // Softmax
            softmax(att, pos + 1);
            // 加权求和
            for (int t = 0; t <= pos; t++) {
                xb += att[t] * v[t];
            }
        }
        
        // 2.5 输出投影
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);
        
        // 2.6 残差连接
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }
        
        // 2.7 FFN 前的 RMSNorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);
        
        // 2.8 FFN（SwiGLU）
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        // SwiGLU 激活
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = silu(s->hb[i]) * s->hb2[i];
        }
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
        
        // 2.9 残差连接
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }
    
    // 3. 最终归一化
    rmsnorm(x, x, w->rms_final_weight, dim);
    
    // 4. 输出分类器
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    
    return s->logits;
}
```


---

## 分词器：BPE

### BPE（Byte Pair Encoding）

BPE 是一种子词分词算法，它让模型能够自适应地处理文本：

- **常见词**（如 "the"）：直接合并成一个 Token，节省计算量
- **罕见词**（如 "Transformer"）：可能拆成 "Trans", "former" 两个 Token
- **从未见过的词**：拆成一个个字节

### 编码流程

假设你输入 "hi"：

1. **准备**：加上 `<s>` 和前置空格 " hi"
2. **原子化**：拆成 `[空格, h, i]`
3. **寻找合并**：
   - 检查 `(空格, h)`，发现词典里有 " h"，分数 10.5
   - 检查 `(h, i)`，发现词典里有 "hi"，分数 12.0
4. **执行合并**：因为 "hi" 分数更高，先合并后面，变成 `[空格, hi]`
5. **再次合并**：检查 `(空格, hi)`，如果词典里有 " hi" 且分数够高，合并成 `[ hi]`
6. **输出**：最后得到一个 ID

### 为什么要加前置空格？

在英语中，绝大多数单词在句子中间出现时，前面都是带空格的（例如："I like apples"）。因此，在模型训练时，词典里存储的 Token 大多是带前置空格的，比如：

- Token A: " apple" (ID: 12345)
- Token B: "apple" (ID: 6789) —— 这通常被视为一个完全不同的词

这是一个 Llama/SentencePiece 的特殊设计。它会在你的输入前面强行加一个空格。这样处理是为了让 "hello" 出现在句首和句中时，编码结果保持一致。

### UTF-8 处理

BPE 需要正确处理 UTF-8 编码的多字节字符。UTF-8 编码规则：

| 码点范围           | 字节 1   | 字节 2   | 字节 3   | 字节 4   |
| ------------------ | -------- | -------- | -------- | -------- |
| U+0000 - U+007F    | 0xxxxxxx |          |          |          |
| U+0080 - U+07FF    | 110xxxxx | 10xxxxxx |          |          |
| U+0800 - U+FFFF    | 1110xxxx | 10xxxxxx | 10xxxxxx |          |
| U+10000 - U+10FFFF | 11110xxx | 10xxxxxx | 10xxxxxx | 10xxxxxx |

**关键判断**：

- `(*c & 0xC0) != 0x80`：判断当前字节是不是 UTF-8 的起始字节
- `0xC0` 是 `11000000`，`0x80` 是 `10000000`
- 在 UTF-8 中，所有后续字节都以 "10" 开头

### 编码代码流程

```c
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, 
            int *tokens, int *n_tokens) {
    // 1. 添加 BOS token（如果需要）
    if (bos) tokens[(*n_tokens)++] = 1;
    
    // 2. 添加虚拟前置空格
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }
    
    // 3. UTF-8 初步拆解（原子化）
    for (char *c = text; *c != '\0'; c++) {
        // 判断是否为起始字节
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        
        // 继续读取后续字节
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }
        
        // 读完一个完整字符，查表
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            // 字节回退：如果找不到，就按字节编码
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }
    
    // 4. BPE 合并循环
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        
        // 扫描所有相邻 token 对
        for (int i=0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", 
                    t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        
        if (best_idx == -1) break; // 没有可合并的了
        
        // 合并最佳对
        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }
    
    // 5. 添加 EOS token（如果需要）
    if (eos) tokens[(*n_tokens)++] = 2;
}
```

### 解码流程

解码相对简单，就是将 Token ID 映射回字符串：

```c
char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // BOS 后的空格处理
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // 处理字节 token（如 <0x01>）
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}
```

### 完整流程示意

```
编码 (Encoding): "Hello" → [15496]
    ↓
推理 (Inference): 模型处理 [15496] → 预测 [13]（对应 ","）
    ↓
解码 (Decoding): [13] → ","
    ↓
展示: safe_printf 打印 ","
```


---

## 采样策略

如果说 `forward`（前向传播）是 Transformer 的"思考"过程，那么 **Sampler（采样器）**就是它的"决策过程"。

在 `forward` 结束时，模型给出的不是一个确定的词，而是对词典中所有词（通常是 32,000 个）的"打分"（Logits）。采样器的任务就是根据这些分数和一些参数，最终拍板决定：**下一个词到底选谁？**

### 三种采样风格

| 风格                     | 参数条件              | 描述                                | 效果                           |
| ------------------------ | --------------------- | ----------------------------------- | ------------------------------ |
| 贪心采样 (Greedy)        | `temp == 0`           | 永远选概率最高的那一个              | 稳定、死板、容易循环           |
| 多项式采样 (Multinomial) | `temp > 0, topp == 1` | 按照概率分布"抽奖"                  | 有创意，但也可能胡言乱语       |
| 核采样 (Top-P/Nucleus)   | `0 < topp < 1`        | 只在概率加和达到 P 的"头部"词中抽奖 | 主流方案：平衡了多样性与逻辑性 |

### 1. 贪心采样（Greedy Sampling）

永远选择概率最高的那个词。

```c
int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}
```

**优点**：稳定、可复现
**缺点**：死板、容易陷入重复循环

### 2. 多项式采样（Multinomial Sampling）

按照概率分布随机抽取。

```c
int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}
```

**工作原理**：

- 从头依次累加概率
- 当累加值超过随机数 `coin`（0-1 之间）时，返回当前 token

**优点**：有创意、多样性高
**缺点**：可能选到低概率的"胡言乱语"

### 3. 核采样（Top-P / Nucleus Sampling）

只在概率加和达到 P（如 0.9）的"核心"词中抽奖。

```c
int sample_topp(float* probabilities, int n, float topp, 
                ProbIndex* probindex, float coin) {
    // 1. 预筛选：剔除极低概率的词
    const float cutoff = (1.0f - topp) / (n - 1);
    int n0 = 0;
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    
    // 2. 排序：按概率从高到低
    qsort(probindex, n0, sizeof(ProbIndex), compare);
    
    // 3. 截断：累加到 topp 为止
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }
    
    // 4. 在截断后的列表中采样
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}
```

**工作流程**：

1. **预筛选**：为了效率，先剔除掉概率极低的词（cutoff 逻辑）
2. **排序**：将候选词按概率从高到低排列
3. **截断**：从高到低累加概率，一旦总和达到 `topp`（例如 0.9），剩下的词全部扔掉
4. **再采样**：在剩下的这几个高概率词（"核"）中重新抽奖

**优点**：平衡了多样性与逻辑性，是目前主流方案

### Temperature（温度）参数

Temperature 控制概率分布的"陡峭程度"：

```c
// 应用温度
for (int q=0; q<sampler->vocab_size; q++) { 
    logits[q] /= sampler->temperature; 
}
// 再应用 softmax
softmax(logits, sampler->vocab_size);
```

**效果**：

- `temp = 0`：贪心模式，永远选最高概率
- `temp = 1`：保持原始分布
- `temp < 1`（如 0.5）：分布更陡峭，更倾向于高概率词（更保守）
- `temp > 1`（如 1.5）：分布更平缓，低概率词也有机会（更随机）

### 完整采样流程

```c
int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        // 贪心采样
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // 应用温度
        for (int q=0; q<sampler->vocab_size; q++) { 
            logits[q] /= sampler->temperature; 
        }
        // Softmax
        softmax(logits, sampler->vocab_size);
        // 生成随机数
        float coin = random_f32(&sampler->rng_state);
        // 选择采样方式
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, 
                              sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}
```


---

## 推理循环

### Generate 模式：文本续写

这是最基础的生成模式，给定一个 prompt，模型会一个词一个词地续写下去。

```c
void generate(Transformer *transformer, Tokenizer *tokenizer, 
              Sampler *sampler, char *prompt, int steps) {
    // 1. 编码 prompt
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    
    // 2. 初始化
    int token = prompt_tokens[0];
    int pos = 0;
    long start = 0;
    
    // 3. 主循环
    while (pos < steps) {
        // 3.1 前向传播
        float* logits = forward(transformer, token, pos);
        
        // 3.2 决定下一个 token
        int next;
        if (pos < num_prompt_tokens - 1) {
            // Prompt 填充期：强制使用下一个输入词
            next = prompt_tokens[pos + 1];
        } else {
            // 自主生成期：采样选词
            next = sample(sampler, logits);
        }
        pos++;
        
        // 3.3 退出条件
        if (next == 1) { break; }
        
        // 3.4 解码并显示
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout); // 流式输出
        
        // 3.5 更新状态
        token = next;
        
        // 3.6 启动计时器
        if (start == 0) { start = time_in_ms(); }
    }
    
    // 4. 报告性能
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", 
                (pos-1) / (double)(end-start)*1000);
    }
    
    free(prompt_tokens);
}
```

### 生成流程详解

每一步的循环包含：

1. **输入**：上一次生成的 Token（或者 Prompt 的第一个词）
2. **思考**：Transformer 经过上千亿次浮点运算（`forward`）
3. **决策**：采样器在几万个候选词中挑一个（`sample`）
4. **翻译**：分词器把数字变回字符（`decode`）
5. **反馈**：把选中的词塞回模型，回到步骤 1

### 为什么要 fflush(stdout)？

在 C 语言中，输出通常是"行缓冲"的。如果不加这一行，程序会等模型生成完一整行才一股脑显示出来。加上 `fflush` 后，你就能看到模型一个字、一个字蹦出来的效果，这种"流式输出"让 AI 看起来更像是在实时思考。

### Chat 模式：对话引擎

Chat 模式在 Generate 的基础上增加了对话管理和特殊格式处理。

#### Llama 2 Chat 格式

```
[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]
```

**组件说明：**

| 组件          | 对应代码/格式      | 形象比喻          | 核心功能                               |
| ------------- | ------------------ | ----------------- | -------------------------------------- |
| System Prompt | `<<SYS>>` 内部内容 | 底层架构/剧本设定 | 设定 AI 的"人格"、"规则"和"知识边界"   |
| User Prompt   | `[INST]` 里的正文  | 甲方的具体要求    | 用户当前输入的问题或指令               |
| Instruction   | `[INST]` 标签本身  | 工作指令信号灯    | 告诉模型："别再瞎写了，现在开始干活！" |

**示例：**

```
[INST] <<SYS>>
你是一个严谨的法官，只回答法律问题。
<</SYS>>

帮我写一份离婚协议书 [/INST]
```

#### Chat 代码流程

```c
void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    
    int8_t user_turn = 1; // 1=用户输入，0=模型回复
    int pos = 0;
    
    while (pos < steps) {
        if (user_turn) {
            // 1. 获取 system prompt（仅第一次）
            if (pos == 0) {
                if (cli_system_prompt == NULL) {
                    read_stdin("Enter system prompt (optional): ", 
                              system_prompt, sizeof(system_prompt));
                } else {
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            
            // 2. 获取 user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                strcpy(user_prompt, cli_user_prompt);
            } else {
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            
            // 3. 拼接成 Llama 2 Chat 格式
            if (pos == 0 && system_prompt[0] != '\0') {
                sprintf(rendered_prompt, 
                       "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]",
                       system_prompt, user_prompt);
            } else {
                sprintf(rendered_prompt, "[INST] %s [/INST]", user_prompt);
            }
            
            // 4. 编码
            encode(tokenizer, rendered_prompt, 1, 0, 
                  prompt_tokens, &num_prompt_tokens);
            user_idx = 0;
            user_turn = 0;
            printf("Assistant: ");
        }
        
        // 5. 确定输入 token
        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++]; // 填充 KV Cache
        } else {
            token = next;
        }
        
        // 6. EOS (=2) 结束 Assistant 回合
        if (token == 2) { user_turn = 1; }
        
        // 7. 前向传播 + 采样
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;
        
        // 8. 显示 Assistant 的回复
        if (user_idx >= num_prompt_tokens && next != 2) {
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece);
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    
    free(prompt_tokens);
}
```

### 两种模式的区别

| 特性     | Generate 模式      | Chat 模式               |
| -------- | ------------------ | ----------------------- |
| 用途     | 文本续写           | 多轮对话                |
| 输入格式 | 纯文本             | `[INST]...[/INST]` 格式 |
| 状态管理 | 单次生成           | 多轮交互                |
| 结束条件 | 达到 steps 或 BOS  | 达到 steps 或 EOS       |
| 典型应用 | 故事续写、代码补全 | 聊天机器人、问答系统    |


---

## 总结

### Transformer 的核心思想

1. **Self-Attention**：让每个词都能"看到"句子中的其他词，理解上下文
2. **Multi-Head**：从多个角度理解同一个句子
3. **FFN**：对每个词进行深度加工，提取语义特征
4. **残差连接**：保留原始信息，防止梯度消失
5. **层归一化**：稳定训练过程，加速收敛
6. **位置编码**：让模型理解词的顺序关系

### 关键优化技术

#### 1. KV Cache

**问题**：每生成一个新词，都要重新计算之前所有词的 K 和 V，非常浪费。

**解决**：将每一层的 K 和 V 缓存起来，新词只需要计算自己的 K 和 V，然后与缓存的进行注意力计算。

**代价**：显存占用大（可能达到几 GB）。

#### 2. Grouped-Query Attention (GQA)

**问题**：标准的 Multi-Head Attention 中，每个头都有自己的 K 和 V，显存占用巨大。

**解决**：让多个 Q 头共享同一组 K 和 V 头。例如：

- 8 个 Q 头
- 2 个 KV 头
- 每 4 个 Q 头共享 1 组 KV

**效果**：显存占用减少 75%，性能损失很小。

#### 3. RoPE（旋转位置编码）

**优势**：

- 相对位置编码，泛化能力强
- 可以外推到更长的序列
- 计算高效，不需要额外的参数

#### 4. RMSNorm

**相比 LayerNorm 的优势**：

- 不需要计算均值，只计算均方根
- 计算量减少约 50%
- 效果几乎一样

#### 5. SwiGLU 激活函数

**相比 ReLU 的优势**：

- 更平滑的梯度
- 门控机制增强表达能力
- 现代大模型的标准配置

### 内存管理策略

#### Weights（权重）：使用 mmap

```c
*data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
```

**优势**：

- **零拷贝**：权重数据不需要从内核缓冲区拷贝到用户缓冲区
- **延迟加载**：操作系统仅在实际访问某页权重时才将其载入物理内存
- **节省内存**：多个进程可以共享同一份权重文件

#### RunState（激活值）：使用 malloc

```c
s->x = calloc(p->dim, sizeof(float));
```

**原因**：

- 激活值需要频繁读写
- 每个进程都有自己的激活值
- 使用堆内存更灵活

### 模型文件结构

```
[Config 结构体]
[Token Embedding Table]
[Layer 0 RMS Attention Weight]
[Layer 1 RMS Attention Weight]
...
[Layer 0 WQ]
[Layer 1 WQ]
...
[Layer 0 WK]
[Layer 1 WK]
...
[Layer 0 WV]
[Layer 1 WV]
...
[Layer 0 WO]
[Layer 1 WO]
...
[Layer 0 RMS FFN Weight]
[Layer 1 RMS FFN Weight]
...
[Layer 0 W1]
[Layer 1 W1]
...
[Layer 0 W2]
[Layer 1 W2]
...
[Layer 0 W3]
[Layer 1 W3]
...
[Final RMS Weight]
[RoPE Freq (跳过)]
[Classifier Weights (可选)]
```

### 理论到代码的映射

| 理论概念             | 代码实现                                   | 数据结构                           |
| -------------------- | ------------------------------------------ | ---------------------------------- |
| Token Embedding      | `token_embedding_table`                    | `float[vocab_size][dim]`           |
| Query/Key/Value      | `wq`, `wk`, `wv`                           | `float[n_layers][dim][dim]`        |
| Multi-Head Attention | `for (h = 0; h < n_heads; h++)`            | 循环并行                           |
| Attention Score      | `score = dot(q, k) / sqrt(head_size)`      | `float[n_heads][seq_len]`          |
| Softmax              | `softmax(att, pos + 1)`                    | 原地修改                           |
| Weighted Sum         | `xb += att[t] * v[t]`                      | 累加                               |
| FFN                  | `w1`, `w2`, `w3` + SwiGLU                  | `float[n_layers][hidden_dim][dim]` |
| Residual             | `x[i] += xb[i]`                            | 逐元素相加                         |
| LayerNorm            | `rmsnorm(xb, x, weight, dim)`              | 归一化 + 缩放                      |
| RoPE                 | 旋转矩阵应用                               | 原地修改 Q 和 K                    |
| KV Cache             | `key_cache`, `value_cache`                 | `float[n_layers][seq_len][kv_dim]` |
| Logits               | `matmul(logits, x, wcls, dim, vocab_size)` | `float[vocab_size]`                |
| Sampling             | `sample(sampler, logits)`                  | 返回 token ID                      |

