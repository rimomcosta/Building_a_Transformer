## Chapter 17: Putting It All Together (Complete Example)

### The Grand Tour: Following One Sentence Through the Entire Transformer

**Congratulations on making it this far!** You've learned every component. Now let's watch them all work together in one beautiful choreographed dance!

**Think of this like watching a factory assembly line:**
- Raw materials enter (text: "I love pizza")
- Each station adds something (tokenization → embeddings → attention → FFN → ...)
- Final product comes out (prediction: "and")

**Or like watching a seed become a plant:**
- Seed (input text)
- Roots form (tokenization)
- Stem grows (embeddings + position)
- Branches spread (attention - gathering information)
- Leaves process sunlight (FFN - individual processing)
- Multiple growth cycles (stacking layers)
- Fruit appears (final prediction)

Let's trace our complete pipeline with final trained weights! We'll follow "I love pizza" through every single step.

### Input

Sentence: "I love pizza"

### Step-by-Step (Inference Mode)

**1. Tokenization:**
```
"I love pizza" → [123, 567, 999]
```

**2. Embedding Lookup (trained weights):**
```
"I":    [0.15, -0.31, 0.44, 0.28, -0.19, 0.37]
"love": [-0.52, 0.73, 0.11, 0.41, 1.02, -0.44]
"pizza": [1.03, -0.18, -0.33, -0.97, 0.68, 0.91]
```

**3. Add Positional Encoding:**
```
"I":    [0.15, 0.69, 0.44, 1.28, -0.19, 1.37]
"love": [0.32, 1.27, 0.16, 1.41, 1.02, 0.56]
"pizza": [1.94, -0.60, -0.24, 0.03, 0.68, 1.91]
```

**4. Transformer Block 1:**

Multi-head attention → Add & Norm → FFN → Add & Norm

Output:
```
"I":    [0.71, -0.28, 0.95, 0.53, -0.19, 0.44]
"love": [0.48, 0.91, 0.32, 0.77, 0.68, 0.29]
"pizza": [1.12, -0.43, -0.11, 0.25, 0.81, 1.04]
```

**5. Transformer Block 2:**

Output:
```
"I":    [0.88, -0.41, 1.13, 0.67, -0.25, 0.59]
"love": [0.63, 1.08, 0.45, 0.92, 0.81, 0.42]
"pizza": [1.29, -0.57, -0.04, 0.38, 0.94, 1.19]
```

**6. Transformer Block 3:**

Output:
```
"I":    [0.99, -0.51, 1.27, 0.78, -0.31, 0.71]
"love": [0.74, 1.21, 0.56, 1.03, 0.92, 0.53]
"pizza": [1.42, -0.68, 0.08, 0.49, 1.04, 1.31]
```

**7. Final LayerNorm:**
```
"pizza": [1.15, -0.89, 0.14, 0.37, 0.95, 1.08]
```

**8. Vocabulary Projection (50,000 dim):**
```
logits = [1.15, -0.89, ..., 1.08] × W_vocab
→ [0.23, -1.44, ..., 5.67, ..., 0.89]
        ↑
      token 4567 ("delicious") = 5.67 (highest!)
```

**9. Softmax:**
```
P("delicious") = exp(5.67) / sum(all_exps) = 0.427 (42.7%)
```

**10. Sample:**
```
Next token: "delicious" (4567)
```

**11. Repeat for "I love pizza delicious"**, and so on!

---

**Course navigation:** [Previous: Chapter 16 - Common Training Problems & Solutions](chapter-16-common-training-problems-solutions.md) | [Next: Chapter 18 - From Language Model to ChatGPT](chapter-18-from-language-model-to-chatgpt.md)
