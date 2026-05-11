## Chapter 14: All the Hyperparameters (The Control Panel)

### Understanding the Control Panel

**Imagine you're learning to drive a car.** The car has many controls:
- **Steering wheel:** Which direction?
- **Gas pedal:** How fast?
- **Brake pedal:** Slow down?
- **Gear shift:** Power vs efficiency?

**Transformers have similar controls called hyperparameters!** These are the settings you choose BEFORE training starts. They control:
- How big is the model? ($d_{\text{model}}$, number of layers)
- How does it learn? (learning rate, batch size)
- How does it avoid mistakes? (dropout rate)

**Why "hyper" parameters?** Because they're parameters ABOUT the parameters! The weights and embeddings are parameters the model learns. Hyperparameters are settings that control HOW the model learns those weights.

**Think of it like baking:**
- **Ingredients** = the weights (model learns these)
- **Oven temperature, baking time** = hyperparameters (you set these!)

Get the hyperparameters wrong, and even the best ingredients won't make good bread!

Let's see all the dials you can turn:

### Model Architecture Hyperparameters

These control the MODEL SIZE and STRUCTURE:

**1. $d_{\text{model}}$** - Embedding dimension (How rich is each word's representation?)
- Our tutorial: 6
- GPT-2: 768 (small), 1024 (medium), 1280 (large), 1600 (XL)
- GPT-3: 12,288
- Effect: Larger = more capacity, more computation

**2. Number of layers $N$**
- Our tutorial: 3
- GPT-2: 12-48
- GPT-3: 96
- Effect: Deeper = more reasoning, slower inference

**3. Number of attention heads $h$**
- Our tutorial: 2
- GPT-2: 12-25
- GPT-3: 96
- Must divide $d_{\text{model}}$ evenly (so each head gets the same dimension)

**Why multiple heads?** Remember the camera analogy—different heads focus on different aspects of the relationships between words. More heads = more diverse perspectives. But there's a trade-off: if you have 96 heads dividing 12,288 dimensions, each head only gets 128 dimensions to work with (12,288 ÷ 96 = 128). Fewer, richer heads vs. many simpler heads—both work!

**4. Feed-forward dimension $d_{ff}$**
- Typical: $4 \times d_{\text{model}}$
- GPT-3: 49,152
- Effect: Captures more complex patterns

**5. Vocabulary size**
- Our tutorial: 50,000
- GPT-2: 50,257
- GPT-3: 50,257
- Some frontier tokenizers: on the order of 100,000 tokens
- Effect: More tokens = better rare word handling, larger embedding matrix

**6. Context length (max sequence)**
- GPT-2: 1024
- GPT-3: 2048
- Modern frontier models: ranges from thousands to hundreds of thousands of tokens, depending on the system
- Effect: Longer = more context, quadratically more memory

### Training Hyperparameters

**7. Learning rate $\eta$**
- Typical: 0.0001 - 0.0006
- Too high: Training unstable, loss explodes
- Too low: Training too slow, might get stuck
- Often uses warmup + decay schedule

**8. Batch size**
- GPT-3: 3.2 million tokens per batch!
- Trade-off: Larger = more stable, needs more memory
- Effective batch size via gradient accumulation

**9. Dropout rate $p$**
- Typical: 0.1 - 0.3
- Applied after attention and FFN
- Prevents overfitting

**10. Weight decay**
- L2 regularization: 0.01 - 0.1
- Prevents weights from growing too large
- $W_{\text{new}} = W_{\text{old}} - \eta(\nabla L + \lambda W_{\text{old}})$

**11. Gradient clipping**
- Max gradient norm: 1.0
- Prevents exploding gradients
- If $||\nabla|| > 1.0$, scale down: $\nabla \leftarrow \frac{\nabla}{||\nabla||}$

**12. Adam parameters**
- $\beta_1 = 0.9$ (momentum)
- $\beta_2 = 0.999$ (adaptive learning rate)
- $\epsilon = 10^{-8}$

### Learning Rate Schedule

**Warmup + Cosine Decay:**

Instead of a constant learning rate, we vary it during training:

```
Steps 0-4000: Linear warmup from 0 to peak
Step 4000: Peak learning rate (e.g., 0.0006)
Steps 4000-100,000: Cosine decay to 0.00006
```

**Cosine decay formula:**
$$\eta(t) = \eta_{\max} \times 0.5 \times \left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

Where:
- $t$ = current step
- $T$ = total training steps
- $\eta_{\max}$ = peak learning rate

**Why warmup?** 

At the start, weights are random and predictions are nonsense. If we use full learning rate immediately, the huge gradients cause:
- Exploding activations
- Numerical instability (NaN values)
- Catastrophic forgetting of initialization patterns

Warmup is like slowly accelerating a car from 0 to 60 mph instead of flooring it instantly!

**Why decay?**

As training progresses:
- Loss gets smaller (we're near the optimum)
- We want to "fine-tune" with small adjustments
- Large steps might overshoot the minimum

Decay is like slowing down as you approach your parking spot—you don't want to ram the curb!

**Example learning rate over time:**
```
Step 0:     η = 0.0000
Step 1000:  η = 0.00015  (warmup)
Step 2000:  η = 0.00030  (warmup)
Step 4000:  η = 0.00060  (peak!)
Step 20000: η = 0.00055  (decay starting)
Step 50000: η = 0.00030  (decay continuing)
Step 90000: η = 0.00008  (near end, very small)
Step 100000: η = 0.00006  (minimum)
```

---

**Course navigation:** [Previous: Chapter 13 - Inference](chapter-13-inference.md) | [Next: Chapter 15 - Additional Techniques](chapter-15-additional-techniques.md)
