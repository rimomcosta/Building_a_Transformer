## Chapter 15: Additional Techniques

### Gradient Accumulation

**Problem:** Can't fit batch size 32 in GPU memory

Imagine your GPU only has enough memory for 8 examples, but you want the stability of batch size 32.

**Solution:** Accumulate gradients over multiple mini-batches

**Algorithm:**
```
1. Clear gradients to zero
2. For i = 1 to 4:
    a. Process mini-batch of 8 examples
    b. Compute loss and gradients
    c. ADD gradients to accumulator (don't update weights yet!)
3. Divide accumulated gradients by 4 (to average them)
4. Update weights with accumulated gradients
5. Clear accumulated gradients
6. Repeat!
```

**Concrete example:**

Mini-batch 1 (8 examples):
- $\frac{\partial L}{\partial W} = 0.5$

Mini-batch 2 (8 examples):
- $\frac{\partial L}{\partial W} = 0.7$

Mini-batch 3 (8 examples):
- $\frac{\partial L}{\partial W} = 0.3$

Mini-batch 4 (8 examples):
- $\frac{\partial L}{\partial W} = 0.6$

**Accumulated gradient:**
$$\frac{0.5 + 0.7 + 0.3 + 0.6}{4} = \frac{2.1}{4} = 0.525$$

This is mathematically equivalent to processing all 32 examples at once!

Effective batch size = 4 × 8 = 32

**Trade-off:** 4× more forward/backward passes (slower) but fits in memory (essential!)

### Mixed Precision Training

Use 16-bit floats (FP16) instead of 32-bit (FP32):
- **2× faster training** (GPUs have specialized FP16 hardware)
- **2× less memory** (16 bits vs 32 bits per number)
- Requires loss scaling to prevent underflow

**What's the difference?**

**FP32 (Float32):** Standard precision
- Range: $\pm 3.4 \times 10^{38}$ to $\pm 1.4 \times 10^{-45}$
- Can represent very tiny gradients like 0.00000001

**FP16 (Float16):** Half precision
- Range: $\pm 65,504$ to $\pm 6.1 \times 10^{-5}$
- Numbers smaller than ~0.00006 become ZERO (underflow!)

**The problem with gradients:**

During backprop, gradients can be tiny:
```
Layer 96 gradient: 0.00001234  ← FP32: OK ✓
                                   FP16: Becomes 0 ✗ (underflow!)
```

If gradients become zero, learning stops!

**Solution: Loss Scaling**

Multiply loss by a large number (e.g., 1024) before backprop:

```
Original loss: 3.2
Scaled loss:   3.2 × 1024 = 3276.8

Backprop with scaled loss:
Original gradient: 0.00001234
Scaled gradient:   0.00001234 × 1024 = 0.01264  ← FP16: OK ✓

After backprop, divide gradients by 1024:
Final gradient: 0.01264 / 1024 = 0.00001234
```

Now we can represent tiny gradients in FP16!

**Example comparison:**
```
Number          FP32         FP16 (no scaling)    FP16 (with 1024× scaling)
0.00001234      0.00001234   0.0 (underflow!)    0.01264 → 0.00001234 ✓
0.5             0.5          0.5                 512.0 → 0.5 ✓
1234.567        1234.567     1234.5              1,264,197 → 1234.567 ✓
```

**Modern practice:** PyTorch and TensorFlow handle this automatically with "automatic mixed precision" (AMP).

### Gradient Checkpointing

Trade computation for memory:
- Don't store all activations during forward pass
- Recompute them during backward pass as needed
- Enables training larger models on limited hardware

**The memory problem:**

During backpropagation, we need the activations from the forward pass to compute gradients. With 96 layers, storing everything uses enormous memory!

**Normal training (high memory):**
```
Forward pass:
  Layer 1 → activations (SAVE)
  Layer 2 → activations (SAVE)
  ...
  Layer 96 → activations (SAVE)
  
Backward pass:
  Layer 96: Use saved activations ✓
  Layer 95: Use saved activations ✓
  ...

Memory: 96 activation tensors stored!
```

**With checkpointing (low memory):**
```
Forward pass:
  Layer 1 → activations (SAVE - checkpoint)
  Layer 2-23 → activations (DISCARD)
  Layer 24 → activations (SAVE - checkpoint)
  ...
  Layer 96 → activations (SAVE - checkpoint)
  
Backward pass:
  Layer 96: Use saved activations ✓
  Layer 95: Recompute from Layer 72 checkpoint
  Layer 94: Recompute from Layer 72 checkpoint
  ...

Memory: Only ~4 checkpoints stored!
```

**Example:**

Without checkpointing: Store 96 layers × 512 MB each = 49 GB
With checkpointing (every 24 layers): Store 4 checkpoints × 512 MB = 2 GB

**Trade-off:**
- **Memory:** 96× less (good!)
- **Compute:** 2× more (have to recompute activations during backward)

This is often worth it—you can train models that wouldn't fit in memory at all!

### Weight Initialization Strategy

**Xavier/Glorot Initialization:**

$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{d_{\text{in}} + d_{\text{out}}}}\right)$

**Why?** Maintains variance of activations and gradients across layers.

Example for $d_{\text{in}}=6$, $d_{\text{out}}=24$:
$\sigma = \sqrt{\frac{2}{6+24}} = \sqrt{0.0667} = 0.258$

**Bias initialization:** Always zeros

---

