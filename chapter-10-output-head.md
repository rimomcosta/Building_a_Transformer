## Chapter 10: The Output Head (Predicting Next Token)

After all transformer blocks, we have a final 6-dimensional vector for each position. We need to convert this to probabilities over our entire **50,000-word vocabulary**.

### Linear Transformation to Vocabulary

**Weight matrix:** $\mathbf{W}_{\text{vocab}}$ with shape $6 \times 50000$

This is a massive matrix! It transforms our 6-dimensional vector into 50,000 scores, one for each possible next token.

**Weight Tying: A Clever Trick**

Many transformers use something called weight tying, where:
$$\mathbf{W}_{\text{vocab}} = \mathbf{E}^T$$

This means the output vocabulary matrix is just the transpose of the input embedding matrix we defined way back in Chapter 3!

**Why tie weights?**

**1. Saves massive parameters:** Instead of storing two separate matrices (input embeddings: 50,000×6 AND output projection: 6×50,000), we store just one and reuse it. That's 300,000 numbers saved in our tiny model, or 614 million saved in GPT-3!

**2. Symmetric intuition:** Think about it—the input embeddings learn "what does 'pizza' mean as an input?" and the output projection learns "how likely is 'pizza' as an output?" These are related! If "pizza" as an input is represented by vector [0.9, -0.1, -0.2, -0.8, 0.5, 0.75], it makes sense that when we're predicting outputs, a vector close to this should give "pizza" a high score.

**3. Consistency:** It encourages the model to learn consistent representations. The meaning of "pizza" in the input should relate to how we predict "pizza" in the output.

**The Math:**

Input embedding lookup: row 999 of $\mathbf{E}$ gives us the vector for "pizza"

Output scoring: We compute a dot product between our final hidden state and **each column** of $\mathbf{W}_{\text{vocab}}$. Since $\mathbf{W}_{\text{vocab}}$ has shape $6 \times 50000$, each column corresponds to one vocabulary token.

If $\mathbf{W}\_{\text{vocab}} = \mathbf{E}^T$, then each column of $\mathbf{W}_{\text{vocab}}$ is exactly one word embedding from the input table, just transposed into output form. The dot product measures: "How similar is my current state to the embedding of word X?"

Words whose embeddings are similar to the final hidden state get high scores—they're likely next words!

Weight tying is common, but not universal. Different model families make different trade-offs between parameter efficiency and flexibility.

Final vector for "pizza" position: $[0.71, -0.23, 0.84, 0.45, -0.12, 0.56]$

**Vocabulary projection:**
```math
\text{logits} = [0.71, -0.23, 0.84, 0.45, -0.12, 0.56] \times \mathbf{W}_{\text{vocab}}
```

This gives us 50,000 raw scores (logits), one per token:

```
logits[123] ("I") = -2.3
logits[567] ("love") = 1.8
logits[999] ("pizza") = 0.5
logits[1234] ("the") = 3.2
logits[2001] ("cat") = 2.1
logits[3456] ("is") = 4.7
...
```

### Softmax to Probabilities

Apply softmax over all 50,000 logits:

$$P(\text{token } i) = \frac{e^{\text{logit}_i}}{\sum_{j=1}^{50000} e^{\text{logit}_j}}$$

**Calculation example:**

To keep this **paper-doable**, let's temporarily pretend our vocabulary only contains these six example tokens. In the real model, we'd sum across all 50,000 logits. But on paper, that's impossible and not useful. So we'll do a tiny toy softmax you can actually verify by hand.

```math
\begin{align}
e^{-2.3} &= 0.100 \\
e^{1.8} &= 6.050 \\
e^{0.5} &= 1.649 \\
e^{3.2} &= 24.533 \\
e^{2.1} &= 8.166 \\
e^{4.7} &= 109.947 \\
&\vdots
\end{align}
```

Now sum JUST these six toy values:

$$
\text{sum} = 0.100 + 6.050 + 1.649 + 24.533 + 8.166 + 109.947 = 150.445
$$

**Probabilities:**
```
P("I") = 0.100 / 150.445 = 0.0007 (0.07%)
P("love") = 6.050 / 150.445 = 0.0402 (4.02%)
P("pizza") = 1.649 / 150.445 = 0.0110 (1.10%)
P("the") = 24.533 / 150.445 = 0.1631 (16.31%)
P("cat") = 8.166 / 150.445 = 0.0543 (5.43%)
P("is") = 109.947 / 150.445 = 0.7307 (73.07%)
```

These six probabilities now sum to 100% (up to rounding), so you can check the arithmetic on paper.

**Production note:** In a real 50,000-word model, we'd do the same operation over the full vocabulary. The math is identical. The only thing we're shrinking here is the number of tokens so the example stays human-computable.

In this toy example, the model predicts "is" with highest probability (73.07%). In a full 50,000-token vocabulary, the exact percentage would usually be lower because probability mass would be spread across many more options, but the ranking idea is the same.

### Sampling Strategies

**1. Greedy Sampling:** Always pick highest probability
```python
next_token = argmax(probabilities)  # → "is"
```

**2. Temperature Sampling:** Control randomness

### The Personality Dial Analogy

**Temperature is like controlling someone's personality when they speak:**

**Low Temperature (T = 0.2):** The Boring Accountant
```
You: "Tell me about your day"
Response: "I went to work. I ate lunch. I came home."
(Safe, predictable, no creativity)
```

**Medium Temperature (T = 1.0):** Normal Conversation
```
You: "Tell me about your day"
Response: "I had an interesting meeting at work, then grabbed sushi for lunch with Sarah."
(Natural, balanced)
```

**High Temperature (T = 1.5):** The Creative Artist
```
You: "Tell me about your day"
Response: "My day danced through unexpected moments—work felt like jazz, lunch was a symphony of flavors!"
(Creative, surprising, sometimes weird)
```

**Formula:**
$$P_{\text{temp}}(i) = \frac{e^{\text{logit}_i / T}}{\sum_j e^{\text{logit}_j / T}}$$

**What temperature does:**
- $T = 1.0$: Standard probabilities (unchanged)
- $T < 1.0$ (e.g., 0.5): More confident/deterministic ("sharper" distribution) - picks the obvious choice
- $T > 1.0$ (e.g., 1.5): More random/creative ("flatter" distribution) - considers unusual options

**When to use each:**
- **T = 0.1-0.3**: Answering factual questions ("What is the capital of France?" → "Paris")
- **T = 0.7-0.9**: Chatbots, helpful assistants (natural but not too wild)
- **T = 1.0-1.5**: Creative writing, brainstorming (interesting and diverse)
- **T = 2.0+**: Experimental/artistic text (often bizarre but sometimes brilliant)

**Concrete example with original logits:**
```
Original: "is"=4.7, "the"=3.2, "cat"=2.1
```

**With T=0.5 (low temperature, more deterministic):**

Scaled logits: $[4.7/0.5 = 9.4, 3.2/0.5 = 6.4, 2.1/0.5 = 4.2]$

Calculate exponentials:
$$e^{9.4} \approx 12088, \quad e^{6.4} \approx 602, \quad e^{4.2} \approx 67$$

Sum ≈ 12,757

Probabilities:
```
"is":  12088/12757 = 94.8% ← Much more confident!
"the":   602/12757 = 4.7%
"cat":    67/12757 = 0.5%
```

**With T=1.5 (high temperature, more creative):**

Scaled logits: $[4.7/1.5 = 3.13, 3.2/1.5 = 2.13, 2.1/1.5 = 1.4]$

Calculate exponentials:
$$e^{3.13} = 22.9, \quad e^{2.13} = 8.4, \quad e^{1.4} = 4.1$$

Sum = 35.4

Probabilities:
```
"is":  22.9/35.4 = 64.7% ← Less confident
"the": 8.4/35.4  = 23.7% ← Much higher chance!
"cat": 4.1/35.4  = 11.6% ← Also much higher!
```

Temperature is like a "creativity dial"—low for factual responses, high for creative writing!

**3. Top-k Sampling:** Consider only top k tokens

Set $k = 5$ (only keep top 5 choices)

Original probabilities:
```
"is": 73.07%, "the": 16.31%, "cat": 5.43%, "love": 4.02%, "pizza": 1.10%
"I": 0.07%, ... (remaining toy tokens are tiny)
```

**Step 1:** Keep only top 5, set others to 0:
```
"is": 73.07%, "the": 16.31%, "cat": 5.43%, "love": 4.02%, "pizza": 1.10%
(All others): 0%
```

**Step 2:** Renormalize so they sum to 100%:

Sum = 73.07 + 16.31 + 5.43 + 4.02 + 1.10 = 99.93%

New probabilities:
```
"is":    73.07/99.93 = 73.1%
"the":   16.31/99.93 = 16.3%
"cat":    5.43/99.93 = 5.4%
"love":   4.02/99.93 = 4.0%
"pizza":  1.10/99.93 = 1.1%
```

Now sample randomly from these 5! This prevents the model from ever picking weird low-probability tokens.

**4. Nucleus (top-p) Sampling:** Keep top tokens until cumulative probability exceeds p

Set $p = 0.9$ (keep tokens that make up 90% of probability mass)

Original probabilities (sorted):
```
"is": 73.07%  → cumulative: 73.07%
"the": 16.31% → cumulative: 89.38%
"cat": 5.43%  → cumulative: 94.81% ← STOP! Exceeded 90%
```

**Keep all tokens up to and including "cat"** in this toy example. In a real 50,000-token model, top-p usually keeps a much larger set because the probability mass is spread across many more plausible tokens.

Renormalize these 47 tokens to sum to 100%, then sample.

**Why nucleus over top-k?**
- Top-k is rigid: always exactly k tokens, even if 4 are great and 1 is terrible
- Nucleus is adaptive: might keep 10 tokens if they're all good, or 50 if they're all mediocre

**Comparing all four strategies:**

Imagine you're choosing ice cream flavors:

**Greedy sampling:** Always chocolate (highest probability). Boring but consistent!

**Temperature sampling:** With low temperature (T=0.5), you almost always choose chocolate, occasionally vanilla. With high temperature (T=1.5), you're adventurous—chocolate, vanilla, strawberry, mint chip, even rocky road sometimes! More creative and diverse outputs.

**Top-k sampling:** You decide beforehand "I'll only consider my top 5 favorite flavors" and randomly pick from those. This prevents you from ever choosing a flavor you'd dislike, while still allowing variety.

**Nucleus (top-p) sampling:** You think "I want flavors that represent 90% of my cravings." If you really love chocolate (50% of cravings) and vanilla (40%), those two alone hit 90%, so you only choose between them. But if all flavors are equally appealing (each 10%), you'd need all 10 flavors to reach 90%, giving you more options. It adapts to your confidence!

In practice, modern systems like ChatGPT typically combine these: use temperature around 0.7-0.8 (slightly creative) with nucleus sampling p=0.9-0.95 (prevent really weird outputs). This balances diversity with quality.

---

**Course navigation:** [Previous: Chapter 9 - Stacking Transformer Blocks](chapter-09-stacking-transformer-blocks.md) | [Next: Chapter 11 - Training the Transformer](chapter-11-training-the-transformer.md)
