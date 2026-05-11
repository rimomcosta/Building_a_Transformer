## Chapter 13: Inference (Using the Trained Model)

After training, we have a powerful model. Now let's use it to generate text!

### The Story-Writing Process

**Imagine you're writing a story, one word at a time:**

You start with: "I love"

**Step 1: Think about what word comes next**
- You consider all possible words: "pizza", "you", "cats", "swimming", etc.
- Based on context, "pizza" seems most natural
- You write: "pizza"

**Step 2: Now you have more context**: "I love pizza"
- You think: What comes after "I love pizza"?
- Could be "and", "because", "every", ".", etc.
- You choose: "and"

**Step 3: Even more context**: "I love pizza and"
- What comes next? "pasta", "ice cream", "cheese", etc.
- You choose: "ice cream"

**Each word you write gives you MORE context for the next word!** This is called **autoregressive** generation—each output becomes part of the input for the next prediction.

**The model does EXACTLY this:**
1. Start with: "I love"
2. Predict: "pizza" → Add it to the sequence
3. Now have: "I love pizza"
4. Predict: "and" → Add it
5. Now have: "I love pizza and"
6. Predict: "ice cream" → Add it
7. Continue...

**Like building with blocks:** Each block you place provides support for the next block. The structure grows one piece at a time!

### Generation Loop (Autoregressive Decoding)

Input: "I love"
Goal: Generate 5 more words

**What is autoregressive?** The model generates one token at a time, and each new token is added to the input for the next prediction. It's like writing a story where each word influences the next!

**Iteration 1:**

Input tokens: [123, 567] ("I love")

```
Step 1: Tokenize → [123, 567]
Step 2: Embedding + Positional Encoding
Step 3: Pass through all transformer blocks
Step 4: Output layer → 50,000 logits
Step 5: Softmax → probabilities
```

Top predictions:
```
"pizza": 34.5%
"you": 22.1%
"coding": 18.3%
"the": 12.7%
```

Sample: "pizza" (token 999)

**Current sequence:** "I love pizza"

**Iteration 2:**

Input tokens: [123, 567, 999] ("I love pizza")

**Key point:** We process ALL three tokens through the transformer (with causal masking). The model attends to "I", "love", and "pizza" to predict what comes after "pizza".

```
Process through transformer...
Focus on last position's output (position 2)
```

Top predictions:
```
"and": 28.9%
"with": 19.4%
".": 15.7%
"because": 12.3%
```

Sample: "and" (token 2234)

**Current sequence:** "I love pizza and"

**Iteration 3:**

Input tokens: [123, 567, 999, 2234] ("I love pizza and")

**Key point:** Now processing FOUR tokens! Each iteration the sequence gets longer.

```
Process through transformer...
Focus on last position's output (position 3)
```

Top predictions:
```
"pasta": 31.2%
"ice": 18.5%
"burgers": 16.8%
```

Sample: "pasta" (token 3567)

**Current sequence:** "I love pizza and pasta"

**Pattern recognition:** Notice how predictions make sense:
- After "pizza and", food words like "pasta", "burgers" have high probability
- The model learned from training data that these words often appear together!

Continue until:
- Maximum length reached (e.g., 50 tokens)
- End-of-sequence token generated
- User satisfaction!

**Final output:** "I love pizza and pasta very much."

### Key Differences from Training

| Training | Inference |
|----------|-----------|
| Dropout ON (0.1) | Dropout OFF |
| Causal mask ON | Causal mask ON (still needed!) |
| Batch size 32+ | Batch size 1-8 (user queries) |
| Left-to-right causal, but many positions processed in parallel | Left-to-right causal, usually one next-token step at a time |
| Update weights | Frozen weights |
| Compute gradients | No gradients |

### KV Cache Optimization (Critical for Making Generation Fast!)

**This isn't just an "optimization"—it's the fundamental mechanism that makes autoregressive generation computationally feasible!** Without it, ChatGPT would be 50-100× slower!

**Important clarification:** KV cache is an **inference-time** trick. We use it during generation, when the model is producing tokens one by one. We do **not** use KV cache during standard training, where we process full sequences in parallel and run backpropagation across the whole batch.

### The Recomputation Disaster

**Problem:** Recomputing Keys and Values for previous tokens is wasteful!

**Let's understand why this matters.** Remember that in self-attention, every token computes Q (Query), K (Key), and V (Value). 

**Without caching (wasteful - the naive approach):**
```
Iteration 1: "I love"
- Compute Q₁, K₁, V₁ for "I"
- Compute Q₂, K₂, V₂ for "love"
- Run attention

Iteration 2: "I love pizza"
- Compute Q₁, K₁, V₁ for "I" ← REDUNDANT! Already did this
- Compute Q₂, K₂, V₂ for "love" ← REDUNDANT!
- Compute Q₃, K₃, V₃ for "pizza"
- Run attention

Iteration 3: "I love pizza and"
- Compute Q₁, K₁, V₁ for "I" ← REDUNDANT!
- Compute Q₂, K₂, V₂ for "love" ← REDUNDANT!
- Compute Q₃, K₃, V₃ for "pizza" ← REDUNDANT!
- Compute Q₄, K₄, V₄ for "and"
- Run attention
```

We're recomputing the same K and V values over and over!

**With caching (efficient):**
```
Iteration 1: "I love"
- Compute Q₁, K₁, V₁ for "I"
- Compute Q₂, K₂, V₂ for "love"
- Store K₁, V₁, K₂, V₂ in cache
- Run attention

Iteration 2: "I love pizza"
- Load K₁, V₁, K₂, V₂ from cache ← FAST!
- Only compute Q₃, K₃, V₃ for "pizza"
- Store K₃, V₃ in cache
- Run attention

Iteration 3: "I love pizza and"
- Load K₁, V₁, K₂, V₂, K₃, V₃ from cache ← FAST!
- Only compute Q₄, K₄, V₄ for "and"
- Store K₄, V₄ in cache
- Run attention
```

**Why can we cache K and V but not Q?**

During generation, we only care about predicting the NEXT token:
- Q (Query) for the new token: "What information do I need?"
- K (Key) and V (Value) for all previous tokens: Unchanged! They represent what information is available.

The new token's Query attends to all previous tokens' Keys and Values, but those don't change!

**Complexity reduction:**

Without cache:
- Iteration 1: Process 2 tokens
- Iteration 2: Process 3 tokens (reprocessing 2)
- Iteration 3: Process 4 tokens (reprocessing 3)
- Total: 2 + 3 + 4 + ... + n = $O(n^2)$

With cache:
- Iteration 1: Process 2 tokens
- Iteration 2: Process 1 new token
- Iteration 3: Process 1 new token
- Total: 2 + 1 + 1 + ... = $O(n)$

For generating 100 tokens: 5,050 operations → 101 operations (50× speedup!)

### The Computational Complexity Breakthrough

**Without KV cache:**
- Token 1: Do 1 computation
- Token 2: Do 2 computations (attend to 1 previous)
- Token 3: Do 3 computations (attend to 2 previous)
- ...
- Token 100: Do 100 computations

**Total:** $1 + 2 + 3 + ... + 100 = \frac{100 \times 101}{2} = 5,050$ operations

**This is O(n²) complexity** - quadratic! If you generate 1000 tokens, you need 500,000 operations!

**With KV cache:**
- Token 1: Do 1 computation, cache K₁, V₁
- Token 2: Do 1 computation (reuse K₁, V₁), cache K₂, V₂
- Token 3: Do 1 computation (reuse K₁, V₁, K₂, V₂), cache K₃, V₃
- ...
- Token 100: Do 1 computation (reuse all cached)

**Total:** $1 + 1 + 1 + ... + 1 = 100$ operations

**This is O(n) complexity** - linear! For 1000 tokens, you need just 1000 operations!

**The difference:** $5,050 → 101$ operations (50× faster), or for 1000 tokens: $500,000 → 1,000$ (500× faster!)

**This is why KV cache is not optional—it's essential!**

### Why Context Windows Exist (The Memory Limit)

**"Why can't ChatGPT remember my entire conversation history?"**

**The answer: KV cache memory grows with every token!**

**Trade-off:** Memory for speed
- Storing KV cache: $2 \times L \times h \times d_k$ per token
  - 2 = K and V (both need to be cached)
  - L = number of layers (96 for GPT-3)
  - h = number of heads (96 for GPT-3)
  - $d_k$ = dimension per head (128 for GPT-3)

**For GPT-3:**
$$2 \times 96 \times 96 \times 128 = 2,359,296 \text{ numbers per token}$$

At 32-bit floats (4 bytes each): $2.36M \times 4 = 9.4$ MB per token

**For a full context window:**
- 2048 tokens × 9.4 MB = **19.3 GB of KV cache alone!**
- Plus model weights (350 GB)
- Plus activations during forward pass (several GB)
- **Total:** Needs 80+ GB GPU memory!

**This is why context limits exist:**
- GPT-3: 2048 tokens (fits in 80GB GPU)
- Some modern systems: tens of thousands of tokens
- Some frontier systems: hundreds of thousands of tokens, but only with substantial memory engineering

**The memory wall:**
```
Token 1:    9.4 MB
Token 10:   94 MB
Token 100:  940 MB (~1 GB)
Token 1000: 9.4 GB
Token 2048: 19.3 GB ← GPT-3 limit!
Token 10000: 94 GB ← Needs special hardware!
Token 100000: 940 GB ← frontier-scale context windows need serious engineering
```

**The limiting factor for context length is NOT the math (positional encoding works for any length), it's the MEMORY needed to cache Keys and Values!**

**Practical implication:** This is why when you chat with ChatGPT:
- Short conversation: Fast and cheap
- Very long conversation: Slow and expensive (or hits context limit)
- The model literally runs out of memory to remember everything!

**Recent innovations solving this:**
- FlashAttention: More memory-efficient attention computation, especially for handling the big attention matrix during training and long-context processing
- Sparse attention: Don't attend to every token
- Sliding window attention: Only remember last N tokens
- Compression: Summarize old context into fewer tokens

**Quick distinction:** FlashAttention and KV cache solve DIFFERENT bottlenecks. FlashAttention helps when attention over many tokens becomes memory-heavy. KV cache helps when autoregressive generation would otherwise keep recomputing the same old keys and values.

---

**Course navigation:** [Previous: Chapter 12 - Causal Masking](chapter-12-causal-masking.md) | [Next: Chapter 14 - All the Hyperparameters](chapter-14-all-the-hyperparameters.md)
