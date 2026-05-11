## Chapter 12: Causal Masking (No Cheating!)

### The Cheating Problem

**Imagine taking a multiple-choice test** where all the answers are printed at the bottom of the page:

```
Question 1: What is 2 + 2?
Question 2: What is the capital of France?
Question 3: What is H₂O?

Answers: 4, Paris, Water
```

**If you can see all the answers while answering Question 1**, you're not really learning! You're just copying!

**The same problem happens during transformer training:**

During training, the model sees the entire sentence at once:
```
Input: "I love pizza"
```

When the model is trying to predict the word after "love", it can SEE "pizza" right there in the input! This is cheating! The model could learn to just copy the next word instead of truly understanding language patterns.

**In real life (after training), the model WON'T have access to future words:**
```
User types: "I love"
Model must predict: "???" (doesn't know what comes next!)
```

**So during training, we must simulate this constraint!** The model should only use words it has already "seen" to predict the next word—just like how it will work in real usage.

### Real-World Analogy: The Mystery Novel

**Imagine reading a mystery novel:**

- Page 1: "Detective Smith investigated the crime scene"
- Page 2: "She found a suspicious footprint"
- Page 3: "The butler did it!"

**As you read page 1:** You can only use information from page 1 to predict what happens on page 2. You CAN'T flip ahead and see page 3!

**As you read page 2:** Now you can use pages 1 AND 2 to predict page 3, but still can't peek at page 3!

**This is exactly what causal masking enforces:** Each word can only "see" (attend to) words that came before it, never words that come after!

### The Mask Solution

We apply a **causal mask** to attention scores BEFORE softmax:

**Attention score matrix (3×3 for our sentence):**
```
         I    love  pizza
I     [-0.02, -0.01, 0.16]
love  [ 0.08,  0.12, 0.23]
pizza [ 0.15,  0.19, 0.31]
```

**Causal mask (lower triangular):**
```
         I    love  pizza
I     [ 0,   -∞,    -∞  ]
love  [ 0,    0,    -∞  ]
pizza [ 0,    0,     0  ]
```

**Understanding the mask:**
- **Row 1 (I):** Can attend to position 0 (itself), but positions 1 and 2 are blocked (-∞)
- **Row 2 (love):** Can attend to positions 0-1 (I, love), but position 2 is blocked
- **Row 3 (pizza):** Can attend to all positions 0-2 (I, love, pizza)

This creates the "lower triangular" pattern—each word can only look backward, never forward!

**After adding mask:**
```
         I    love  pizza
I     [-0.02, -∞,    -∞  ]
love  [ 0.08, 0.12,  -∞  ]
pizza [ 0.15, 0.19, 0.31]
```

When we apply softmax, $e^{-\infty} = 0$, so:

**For "love" (row 2):**
```math
\begin{align}
e^{0.08} &= 1.083 \\
e^{0.12} &= 1.128 \\
e^{-\infty} &= 0 \\
\text{Sum} &= 2.211 \\
P(\text{love} \to I) &= 1.083/2.211 = 0.490 \\
P(\text{love} \to \text{love}) &= 1.128/2.211 = 0.510 \\
P(\text{love} \to \text{pizza}) &= 0/2.211 = 0 \quad \text{← Blocked!}
\end{align}
```

"love" can only attend to "I" and itself, not future words!

Think of reading a mystery novel where someone has covered all the pages ahead with sticky notes. You can only use clues from the pages you've already read to make predictions about what happens next. You can't peek ahead! This forces the model to make predictions based solely on past context, just like a human reader would.

Or imagine writing an essay exam where you must answer questions in order, and you're not allowed to look at future questions. Each answer can only use information from previous questions you've already seen. That's exactly what causal masking enforces.

**Implementation:**
```python
mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
scores = scores + mask  # Before softmax
```

---

**Course navigation:** [Previous: Chapter 11 - Training the Transformer](chapter-11-training-the-transformer.md) | [Next: Chapter 13 - Inference](chapter-13-inference.md)
