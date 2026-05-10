## Chapter 5: Multi-Head Self-Attention (The Heart)

This is where the magic happens! This is the innovation that made transformers revolutionary. After reading this chapter, you'll understand how words "talk" to each other and decide who's important.

## Part 1: Understanding the Core Concept

### The Communication Problem

We now have three position-aware word vectors for "I love pizza":
```
"I":     [0.01, 0.80, 0.30, 1.15, -0.05, 1.11]
"love":  [0.44, 1.14, 0.05, 1.25,  0.90, 0.70]
"pizza": [1.81, -0.52, -0.11, 0.20,  0.50, 1.75]
```

**But there's still a problem!** Each word only knows about ITSELF. The vector for "I" contains information about the word "I" and its position, but it doesn't know anything about "love" or "pizza". They're like people standing next to each other but not talking!

For true understanding, words need to LOOK AT each other and exchange information:
- "I" needs to understand what action it's performing ("love")
- "love" needs to know who's doing the loving ("I") and what's being loved ("pizza")
- "pizza" needs to know it's the object being loved

**This is what attention does:** It allows words to communicate and update their representations based on context!

### Real-World Examples of Why This Matters

**Example 1: Pronoun Resolution**
```
"The cat slept because it was tired."
```

The word "it" needs to figure out: "What does 'it' refer to?" By looking at all other words:
- "The" — not a referent
- "cat" — YES! This is a noun that "it" likely refers to
- "slept" — a verb, not a referent
- "because" — a conjunction, not a referent
- "was" — a verb, not a referent
- "tired" — an adjective, not a referent

Through attention, "it" learns to pay most attention to "cat"!

**Example 2: Understanding Relationships**
```
"The chef cooked the pasta."
```

- "chef" needs to know it's the **doer** (subject)
- "cooked" needs to know who's doing it ("chef") and what's being cooked ("pasta")
- "pasta" needs to know it's the **thing being acted upon** (object)

Attention allows each word to gather this contextual information!

**Example 3: Modifier Relationships**
```
"The delicious homemade pizza"
```

- "pizza" needs to know that "delicious" and "homemade" are describing IT
- "delicious" and "homemade" need to know they're modifying "pizza"

Without attention, these words are isolated islands. With attention, they form a connected understanding!

> **Author's note:** In this chapter, we're temporarily letting words look across the whole sentence so you can understand how attention works mathematically. In **Chapter 12**, we'll add **causal masking**, which is the decoder-only rule that prevents a token from looking into the future.

### The Dating App Analogy (Understanding Weighted Attention)

Before diving into math, let's use a really intuitive analogy that shows EXACTLY how attention works: a dating app!

**Imagine you're on a dating app looking for a match.** Each profile has:
- A **headline/bio** (the Key) - what they advertise about themselves
- Their **full profile** (the Value) - all their actual information
- Your **preferences** (the Query) - what you're looking for

**Let's say you're looking for someone who:**
- Likes pizza (very important to you!)
- Wants to get married someday (EXTREMELY important - deal breaker!)
- Enjoys hiking (nice to have)

**You see three profiles:**

**Profile A: Alex**
- Headline (Key): "Pizza lover, enjoys hiking, wants marriage soon"
- Your compatibility score with headline: 95% (matches almost everything!)
- Full profile (Value): [Detailed info about Alex's life, interests, personality]

**Profile B: Blake**
- Headline (Key): "Pizza enthusiast, loves adventure sports, not interested in marriage"
- Your compatibility score with headline: 70% (matches some things)
- Full profile (Value): [Detailed info about Blake's life, interests, personality]

**Profile C: Casey**
- Headline (Key): "Prefers sushi, wants to settle down and marry"
- Your compatibility score with headline: 45% (matches marriage but not food)
- Full profile (Value): [Detailed info about Casey's life, interests, personality]

**Now here's the KEY insight about weighted attention:**

**Step 1: Compute compatibility scores** (how well do their Keys match your Query?)
```
Alex:  95% - loves pizza ✓, hiking ✓, wants marriage ✓
Blake: 70% - loves pizza ✓, adventure ✓, no marriage ✗
Casey: 45% - likes different food ✗, wants marriage ✓
```

**Step 2: But wait! Not all preferences are equally important to you!**

You assign weights to your preferences:
- Likes pizza: weight = 1.0 (important)
- Wants marriage: weight = 10.0 (SUPER important - deal breaker!)
- Enjoys hiking: weight = 0.5 (nice to have)

**Step 3: Recalculate with weighted preferences**

**Alex:**
```
Pizza: YES (1.0 × high score)
Marriage: YES (10.0 × high score) ← HUGE boost!
Hiking: YES (0.5 × high score)
Weighted score: 95 (very high!)
```

**Blake:**
```
Pizza: YES (1.0 × high score)
Marriage: NO (10.0 × zero!) ← HUGE penalty!
Adventure: YES (0.5 × high score)
Weighted score: 15 (dramatically lower!)
```

**Casey:**
```
Pizza: NO (1.0 × low score)
Marriage: YES (10.0 × high score) ← HUGE boost!
Hiking: NO (0.5 × low score)
Weighted score: 55 (moderate)
```

**Final attention distribution** (after softmax):
```
Alex:  80% ← You'll spend most time reading this profile!
Casey: 18% ← Some attention here
Blake: 2%  ← Barely any attention
```

**Step 4: Extract information (weighted combination of Values)**

You don't just read one profile - you read ALL profiles, but spend time proportional to attention:
```
What you learn = 
  80% × Alex's full profile +
  18% × Casey's full profile +
  2% × Blake's full profile
```

You'll learn MOSTLY about Alex (80%), a bit about Casey (18%), and barely anything about Blake (2%).

**THIS IS EXACTLY WHAT ATTENTION DOES!**

- **Query**: What information is this word looking for?
- **Key**: How does each word advertise itself?
- **Scores**: Compatibility between Query and each Key
- **Weights**: Some features matter more than others!
- **Values**: The actual information each word contains
- **Output**: Weighted combination - mostly influenced by high-attention words

### The Asymmetric Attention: The Celebrity-Fan Relationship

**Here's another crucial insight:** Attention is **directional** (not symmetric)!

**Example:** You're a huge Beyoncé fan.

**Your perspective:**
- Your Query: "I want to know everything about Beyoncé!"
- You see Beyoncé's Key: "I'm Beyoncé - singer, performer, icon"
- Your attention score to Beyoncé: 99% (you're obsessed!)
- You extract information from Beyoncé's Value: [All her music, style, achievements]

**You attend to Beyoncé with 99% attention!**

**But from Beyoncé's perspective:**
- Her Query: "Who are the important people in my professional circle?"
- She sees your Key: "I'm a fan from nowhere"
- Her attention score to you: 0.001% (she has millions of fans!)
- She barely notices you exist

**She attends to you with 0.001% attention!**

**The matrix is ASYMMETRIC:**
```
         Beyoncé    You
You      [99%,      1%    ] ← You attend mostly to Beyoncé
Beyoncé  [0.001%,   99.999%] ← She attends mostly to herself/important people
```

**In transformers, the same thing happens!**

In "The famous chef cooked pasta":
- "chef" might attend strongly to "famous" (it's describing me!)
- But "famous" might attend more to "chef" than vice versa
- They don't pay equal attention to each other!

Each word has its own Query (what it's looking for), so attention is directional!

### The Library Analogy (Another Perspective)

Let's use one more analogy to cement the concept. Imagine you're at a library researching Italian cooking.

**You (the Query):** "I'm looking for information about Italian cooking."

You walk past shelves. Each book has a title visible on its spine (the Key):
- Book 1: "Italian Pasta Recipes" ← **Very relevant!** Your query matches this key strongly.
- Book 2: "French Wine Guide" ← Somewhat relevant (both about food, both European)
- Book 3: "Quantum Physics" ← Not relevant at all
- Book 4: "Italian Travel Guide" ← Moderately relevant (Italian, but not cooking)
- Book 5: "Mexican Cooking" ← Moderately relevant (cooking, but not Italian)

**Your attention naturally focuses on Book 1** because its title (Key) matches your interest (Query) most closely!

You pull out Book 1 and read its content (the Value). You might also glance at Books 2, 4, and 5 for supplementary information, but spend most time on Book 1.

**The key insight:** The actual information you extract (Value) comes from the books you paid attention to, based on how well their title (Key) matched what you wanted (Query).

**Now here's the beautiful part:** EVERY word in the sentence does this simultaneously!

In "I love pizza":
- "I" asks: "Who or what am I related to in this sentence?"
- "love" asks: "What's the subject doing this action? What's the object receiving it?"
- "pizza" asks: "What action is being done to me? Who's doing it?"

Each word looks at ALL other words (including itself!) and decides how much attention to pay to each one!

### The Three Components: Query, Key, Value

Let's break down these three mysterious terms:

**Query (Q): "What am I looking for?"**
- This is the question each word asks
- It's like a search query you type into Google
- Example: When "it" in "The cat slept because it was tired" creates its Query, it's asking "What noun do I refer to?"

**Key (K): "What information do I offer?"**
- This is how each word advertises itself
- It's like the title of a book or a headline
- Example: "cat" broadcasts Key information saying "I'm a noun, I'm an animal, I could be a referent"

**Value (V): "What is my actual content?"**
- This is the actual information a word contains
- It's like the content inside the book
- Example: "cat" has Value information containing its full semantic meaning

**Critical insight:** Query, Key, and Value are ALL derived from the same word embedding, but through DIFFERENT transformations (different weight matrices). It's like looking at the same word through three different lenses:
- Q lens: "What does this word need?"
- K lens: "What does this word advertise?"
- V lens: "What information does this word contain?"

### The Attention Process (High-Level Overview)

Here's what happens for EACH word (we'll use "I" as example):

**Step 1: Create Q, K, V for all words**
```
"I" creates:     Q_I, K_I, V_I
"love" creates:  Q_love, K_love, V_love
"pizza" creates: Q_pizza, K_pizza, V_pizza
```

**Step 2: "I" computes attention scores**
```
"I" compares its Query with everyone's Keys:
- How similar is Q_I to K_I? (How much should I attend to myself?)
- How similar is Q_I to K_love? (How much should I attend to "love"?)
- How similar is Q_I to K_pizza? (How much should I attend to "pizza"?)
```

**Step 3: Convert scores to probabilities**
```
Raw scores: [0.15, 0.42, 0.89]
After softmax: [19%, 31%, 50%]  ← These sum to 100%!
```

**Step 4: Create weighted combination of Values**
```
New representation of "I" = 
  19% × V_I + 31% × V_love + 50% × V_pizza
```

**Result:** The word "I" now has an UPDATED representation that incorporates information from all words (but mostly from "pizza" since it got 50% attention)!

**And here's the magic:** This happens for ALL words SIMULTANEOUSLY!
- "I" updates by looking at everyone
- "love" updates by looking at everyone  
- "pizza" updates by looking at everyone

All at the same time, in parallel! That's why transformers are so fast!

---

## Part 2: Multi-Head Attention (Why Multiple Perspectives?)

Before we dive into calculations, we need to understand a crucial design decision: **Why do we have multiple attention heads?**

### The Problem with Single-Head Attention

Imagine if you only had ONE attention mechanism. That single mechanism would have to learn:
- Grammatical relationships (subject-verb-object)
- Semantic relationships (synonyms, antonyms)
- Long-range dependencies (pronoun references)
- Modifier relationships (adjectives to nouns)
- Temporal relationships (before/after)
- Causal relationships (because/therefore)
- ... and many more!

That's asking one mechanism to do TOO MUCH! It would be like having one employee responsible for sales, marketing, engineering, customer support, and accounting all at once!

### The Solution: Multiple Attention Heads

**Key insight:** Different attention heads can specialize in different types of relationships!

**In our example** ($d_{\text{model}} = 6$, 2 heads):
- **Head 1** might learn: "Connect subjects to their verbs and objects" (grammatical structure)
- **Head 2** might learn: "Connect modifiers to what they modify" (descriptive relationships)

**In GPT-3** (96 heads!):
- Head 1 might focus on: Pronouns finding their referents
- Head 2 might focus on: Adjectives finding nouns
- Head 3 might focus on: Verbs finding subjects
- Head 4 might focus on: Understanding negation
- Head 5 might focus on: Numerical relationships
- ...and 91 more specialized patterns!

**The analogy:** Think of a news team covering a story:
- Camera 1: Wide shot showing the big picture
- Camera 2: Close-up of the main speaker
- Camera 3: Audience reactions
- Camera 4: Background context

Each camera has a different perspective. The final news segment combines all perspectives for complete understanding!

### How Multi-Head Attention Works

**Hyperparameters for our example:**
- **Number of heads:** $h = 2$ (GPT-3 uses 96)
- **Head dimension:** $d_k = d_{\text{model}} / h = 6 / 2 = 3$

**Each head works independently on different dimensions:**
- **Head 1:** Uses dimensions [0, 1, 2] of each word
- **Head 2:** Uses dimensions [3, 4, 5] of each word

**Why split dimensions?** This forces each head to use different information! Head 1 looks at one aspect of the word meanings (dimensions 0-2), Head 2 looks at different aspects (dimensions 3-5). This encourages diversity in what they learn!

**Dividing our vectors:**

```
"I":     [0.01, 0.80, 0.30, | 1.15, -0.05, 1.11]
          ︸─────────────────︷  ︸──────────────────︷
              Head 1                 Head 2

"love":  [0.44, 1.14, 0.05, | 1.25,  0.90, 0.70]
          ︸─────────────────︷  ︸──────────────────︷
              Head 1                 Head 2

"pizza": [1.81, -0.52, -0.11, | 0.20,  0.50, 1.75]
          ︸──────────────────︷  ︸──────────────────︷
              Head 1                 Head 2
```

**The complete process:**
1. Head 1 computes attention using dimensions [0,1,2] → produces 3D output for each word
2. Head 2 computes attention using dimensions [3,4,5] → produces 3D output for each word
3. **Concatenate**: Stick the outputs back together → 6D output for each word
4. **Project**: Mix the concatenated heads through a final transformation

**Why is this brilliant?**
- ✓ Each head can specialize
- ✓ Forces diversity (different input dimensions)
- ✓ Still combines into one coherent representation
- ✓ Parallelizable (all heads compute simultaneously)

**🎓 Computational Efficiency Note:**

An important detail: Having $h=2$ heads of size $d_k=3$ (total 6 dimensions) uses roughly the SAME computation and parameters as having 1 head of size 6!

Here's why:
- **Single head** (6D): Each attention needs 6×6 weight matrices for Q, K, V
  - Parameters: 6×6 + 6×6 + 6×6 = 108
  
- **Two heads** (3D each): Each head needs 3×3 weight matrices
  - Parameters per head: 3×3 + 3×3 + 3×3 = 27
  - Total for 2 heads: 27 × 2 = 54
  - Plus output projection: 6×6 = 36
  - Total: 54 + 36 = 90 (approximately the same!)

**The key insight:** Multi-head attention is NOT $h$ times more expensive! It's roughly the same cost as single-head, but split into $h$ parallel specialized pathways. We get multiple perspectives "for free" (no significant extra computation)!

This is why having 96 heads in GPT-3 doesn't make it 96× slower than having 1 head—it's the same total computation, just organized differently.

---

## Part 3: The Mathematics (Step-by-Step Calculations)

Now let's compute everything with actual numbers! We'll focus on **Head 1** and show complete calculations. Head 2 works identically but with dimensions [3,4,5].

**Reading tip:** If you want the intuition first, you can skim the raw arithmetic, look at the final numbers, and then come back with a calculator. The point of this section is that every step is checkable, not that you must do every multiplication in one sitting.

### Step 1: Understand the Weight Matrices

We need three weight matrices per head: $\mathbf{W}^Q$, $\mathbf{W}^K$, $\mathbf{W}^V$

**Important:** These start **random** but follow a specific initialization strategy (Xavier/Glorot):

$$\text{weights} \sim \mathcal{N}\left(0, \frac{2}{d_{\text{in}} + d_{\text{out}}}\right)$$

**What does this notation mean?**
- $\sim$ means "sampled from" or "drawn from"
- $\mathcal{N}(\mu, \sigma^2)$ means a Normal (Gaussian) distribution with mean $\mu$ and variance $\sigma^2$
- The bell curve you learned in statistics!

**Why this specific formula?**

If we initialize with numbers too large (like all 10.0), signals explode through layers. Too small (like all 0.001), signals vanish. Xavier initialization is the "Goldilocks" formula that keeps signal strength roughly constant through layers.

For our $3 \times 3$ matrices: 
$$\sigma = \sqrt{\frac{2}{3+3}} = \sqrt{\frac{2}{6}} = \sqrt{0.333} = 0.577$$

So we sample each weight from a normal distribution centered at 0 with standard deviation 0.577.

**Example of sampling:**
- Random sample 1: 0.2
- Random sample 2: -0.3
- Random sample 3: 0.15
- ... and so on for all weights

But for simplicity in our hand calculations, let's use these small random values:

**Head 1 Weight Matrices (each $3 \times 3$):**

$$
\mathbf{W}^Q_1 = \begin{bmatrix}
0.2 & 0.3 & -0.1 \\
0.1 & -0.2 & 0.4 \\
-0.3 & 0.1 & 0.2
\end{bmatrix}, \quad
\mathbf{W}^K_1 = \begin{bmatrix}
0.1 & -0.2 & 0.3 \\
0.2 & 0.3 & -0.1 \\
0.4 & 0.1 & 0.2
\end{bmatrix}, \quad
\mathbf{W}^V_1 = \begin{bmatrix}
0.3 & 0.1 & 0.2 \\
-0.2 & 0.3 & 0.1 \\
0.1 & -0.1 & 0.3
\end{bmatrix}
$$

### Computing Q, K, V

We multiply each word's embedding (first 3 dims only for Head 1) by these weights.

**For "I" $[0.01, 0.80, 0.30]$:**

Query calculation:
```math
\begin{align}
\mathbf{Q}_I &= [0.01, 0.80, 0.30] \times \mathbf{W}^Q_1 \\
&= \begin{bmatrix}
0.01 \times 0.2 + 0.80 \times 0.1 + 0.30 \times (-0.3) \\
0.01 \times 0.3 + 0.80 \times (-0.2) + 0.30 \times 0.1 \\
0.01 \times (-0.1) + 0.80 \times 0.4 + 0.30 \times 0.2
\end{bmatrix}^T \\
&= \begin{bmatrix}
0.002 + 0.08 - 0.09 \\
0.003 - 0.16 + 0.03 \\
-0.001 + 0.32 + 0.06
\end{bmatrix}^T \\
&= [-0.008, -0.127, 0.379]
\end{align}
```

Key calculation:
```math
\begin{align}
\mathbf{K}_I &= [0.01, 0.80, 0.30] \times \mathbf{W}^K_1 \\
&= [0.001 + 0.16 + 0.12, -0.002 + 0.24 + 0.03, 0.003 - 0.08 + 0.06] \\
&= [0.281, 0.268, -0.017]
\end{align}
```

Value calculation:
```math
\begin{align}
\mathbf{V}_I &= [0.01, 0.80, 0.30] \times \mathbf{W}^V_1 \\
&= [0.003 - 0.16 + 0.03, 0.001 + 0.24 - 0.03, 0.002 + 0.08 + 0.09] \\
&= [-0.127, 0.211, 0.172]
\end{align}
```

**For "love" $[0.44, 1.14, 0.05]$:**

Using the same multiply-and-add method as above:

$$
\begin{align}
\mathbf{Q}_{\text{love}} &= [0.088 + 0.114 - 0.015, 0.132 - 0.228 + 0.005, -0.044 + 0.456 + 0.010] \\
&= [0.187, -0.091, 0.422]
\end{align}
$$

$$\mathbf{K}_{\text{love}} = [0.044 + 0.228 + 0.020, -0.088 + 0.342 + 0.005, 0.132 - 0.114 + 0.010] = [0.292, 0.259, 0.028]$$

$$\mathbf{V}_{\text{love}} = [0.132 - 0.228 + 0.005, 0.044 + 0.342 - 0.005, 0.044 - 0.114 + 0.015] = [-0.091, 0.381, -0.055]$$

**For "pizza" $[1.81, -0.52, -0.11]$:**

$$\mathbf{Q}_{\text{pizza}} = [0.362 - 0.052 + 0.033, 0.543 + 0.104 - 0.011, -0.181 - 0.208 - 0.022] = [0.343, 0.636, -0.411]$$

$$\mathbf{K}_{\text{pizza}} = [0.181 - 0.104 - 0.044, -0.362 - 0.156 - 0.011, 0.543 + 0.052 - 0.022] = [0.033, -0.529, 0.573]$$

$$\mathbf{V}_{\text{pizza}} = [0.543 + 0.104 - 0.011, 0.181 - 0.156 + 0.011, 0.362 - 0.052 - 0.033] = [0.636, 0.036, 0.277]$$

---

## Part 4: Computing Similarity (The Dot Product)

Now we have Q, K, V for all words. The next step is: **How do we measure similarity between a Query and a Key?**

### Understanding Similarity in Everyday Life

**Before we talk about dot products**, let's think about how we measure "similarity" in real life:

**Example 1: Food preferences**

You like: [Pizza: 10/10, Sushi: 7/10, Broccoli: 2/10]
Friend A likes: [Pizza: 9/10, Sushi: 8/10, Broccoli: 1/10]

How similar are you? Very! You both love pizza and sushi, hate broccoli.

Friend B likes: [Pizza: 1/10, Sushi: 2/10, Broccoli: 10/10]

How similar are you? Not at all! You have opposite tastes!

**How would you calculate similarity numerically?**

**Intuitive approach:** Compare each category and add them up:
```
You vs Friend A:
Pizza: (10 × 9) = 90     ← Both love it!
Sushi: (7 × 8) = 56      ← Both like it!
Broccoli: (2 × 1) = 2    ← Both dislike it
Total: 90 + 56 + 2 = 148 (HIGH similarity!)

You vs Friend B:
Pizza: (10 × 1) = 10     ← You love it, they don't
Sushi: (7 × 2) = 14      ← You like it, they don't
Broccoli: (2 × 10) = 20  ← You hate it, they love it!
Total: 10 + 14 + 20 = 44 (LOW similarity)
```

**This multiplication-and-addition is called a DOT PRODUCT!** And it's exactly how we measure Query-Key similarity!

### The Dot Product (Mathematical Definition)

For vectors $\mathbf{a} = [a_1, a_2, a_3]$ and $\mathbf{b} = [b_1, b_2, b_3]$:
$$\mathbf{a} \cdot \mathbf{b} = a_1 \times b_1 + a_2 \times b_2 + a_3 \times b_3$$

**Simple example:**
```math
\begin{align}
[2, 3, 1] \cdot [4, 1, 2] &= (2)(4) + (3)(1) + (1)(2) \\
&= 8 + 3 + 2 \\
&= 13
\end{align}
```

**Another example showing similarity:**
```math
\begin{align}
[1, 2, 3] \cdot [1, 2, 3] &= (1)(1) + (2)(2) + (3)(3) \\
&= 1 + 4 + 9 \\
&= 14 \quad \text{← Identical vectors = high score!}
\end{align}
```

**Opposite vectors:**
```math
\begin{align}
[1, 2, 3] \cdot [-1, -2, -3] &= (1)(-1) + (2)(-2) + (3)(-3) \\
&= -1 - 4 - 9 \\
&= -14 \quad \text{← Opposite vectors = negative score!}
\end{align}
```

**Unrelated vectors:**
```math
\begin{align}
[1, 0, 0] \cdot [0, 1, 0] &= (1)(0) + (0)(1) + (0)(0) \\
&= 0 + 0 + 0 \\
&= 0 \quad \text{← Perpendicular vectors = zero!}
\end{align}
```

**Intuition summary:**
- **Large positive dot product** → vectors point in same direction (very similar!)
- **Near zero** → vectors are perpendicular (unrelated)
- **Large negative** → vectors point opposite directions (opposite meanings)

### Bringing It Back to Our Dating App

Remember our dating app profiles? Let's see this with actual numbers!

**Your preferences (Query):** [Pizza: 10, Marriage: 10, Hiking: 5]

**Alex's profile headline (Key):** [Pizza: 9, Marriage: 9, Hiking: 8]

**Compatibility (dot product):**
```math
\begin{align}
\text{score} &= [10, 10, 5] \cdot [9, 9, 8] \\
&= (10)(9) + (10)(9) + (5)(8) \\
&= 90 + 90 + 40 \\
&= 220 \quad \text{← High compatibility!}
\end{align}
```

**Blake's profile headline (Key):** [Pizza: 9, Marriage: 1, Hiking: 8]

**Compatibility (dot product):**
```math
\begin{align}
\text{score} &= [10, 10, 5] \cdot [9, 1, 8] \\
&= (10)(9) + (10)(1) + (5)(8) \\
&= 90 + 10 + 40 \\
&= 140 \quad \text{← Lower! Marriage mismatch hurt the score!}
\end{align}
```

See how the dot product naturally weights different features? The marriage dimension (10 × 1 = 10) contributed less than the pizza dimension (10 × 9 = 90)!

**This is exactly what happens in attention:** Query·Key computes how compatible/similar the word's question is with what each other word offers!

### Computing Attention Scores for Our Sentence

**Attention scores for "I":**

Dot product $\mathbf{Q}_I \cdot \mathbf{K}_I$:
```math
\begin{align}
\text{score}(I \to I) &= [-0.008, -0.127, 0.379] \cdot [0.281, 0.268, -0.017] \\
&= (-0.008)(0.281) + (-0.127)(0.268) + (0.379)(-0.017) \\
&= -0.0022 - 0.0340 - 0.0064 \\
&= -0.0426
\end{align}
```

Dot product $\mathbf{Q}\_{I} \cdot \mathbf{K}\_{love}$ : 
```math
\begin{align}
\text{score}(I \to \text{love}) &= [-0.008, -0.127, 0.379] \cdot [0.292, 0.259, 0.028] \\
&= -0.0023 - 0.0329 + 0.0106 \\
&= -0.0246
\end{align}
```

Dot product $\mathbf{Q}\_{I} \cdot \mathbf{K}\_{pizza}$:
```math
\begin{align}
\text{score}(I \to \text{pizza}) &= [-0.008, -0.127, 0.379] \cdot [0.033, -0.529, 0.573] \\
&= -0.0003 + 0.0672 + 0.2172 \\
&= 0.2841
\end{align}
```

### Scaling by $\sqrt{d_k}$

**Critical step!** We scale by $\sqrt{d_k} = \sqrt{3} = 1.732$

**Why?** In high dimensions, dot products grow large (variance proportional to $d_k$). Large values fed into softmax create extreme probabilities like $[0.001, 0.001, 0.998]$, which:
- Kills gradients (vanishing gradient problem)
- Makes training unstable
- Prevents the model from learning nuanced attention

The formula:
```math
\text{scaled\_score} = \frac{\mathbf{Q} \cdot \mathbf{K}^T}{\sqrt{d_k}}
```

**Scaled scores for "I":**
```math
\begin{align}
I \to I: & \quad -0.0426 / 1.732 = -0.0246 \\
I \to \text{love}: & \quad -0.0246 / 1.732 = -0.0142 \\
I \to \text{pizza}: & \quad 0.2841 / 1.732 = 0.1640
\end{align}
```

---

## Part 5: Converting Scores to Probabilities (Softmax)

We now have attention scores (similarity measurements), but they're just raw numbers. We need to convert them to percentages that sum to 100%!

### The Cake Sharing Analogy

**Imagine three kids helped bake a cake:**

- Alice helped a lot: contribution score = 10
- Bob helped some: contribution score = 5
- Charlie helped a little: contribution score = 2

**Total help:** 10 + 5 + 2 = 17

**How should we divide the cake fairly?**

```
Alice gets: 10/17 = 59% of the cake (she did the most!)
Bob gets:   5/17 = 29% of the cake
Charlie gets: 2/17 = 12% of the cake

Total: 59% + 29% + 12% = 100% ✓ (whole cake divided!)
```

**This is the BASIC idea of softmax!** Convert raw scores to percentages that sum to 100%.

**But there's a problem with simple division:** What if scores are negative?

```
Attention scores: [-0.02, 0.12, -0.05]

Can't divide negatives into percentages! Alice can't get -15% of cake!
```

**Solution: Softmax uses exponentials to make everything positive!**

### Applying Softmax (Step-by-Step)

Softmax does three things:
1. **Make positive:** Apply exponential ($e^x$) to each score
2. **Sum:** Add all the positive numbers
3. **Normalize:** Divide each by the sum

**Formula:**
$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

**Let's break down the scary notation:**

**$x_i$** = "the i-th value in our list"
- $x_1$ = first score
- $x_2$ = second score  
- $x_3$ = third score

**$e^{x_i}$** = "Euler's number (2.718...) raised to power $x_i$"
- This makes numbers positive
- $e$ is a special mathematical constant (like π)

**$\sum_{j} e^{x_j}$** = "sum up all the exponentials"
- $\sum$ (sigma) means "add up"
- Example: $\sum_{i=1}^{3} x_i = x_1 + x_2 + x_3$

**Simple example to understand $\sum$:**

If $x = [10, 20, 30]$:
$$\sum_{i=1}^{3} x_i = 10 + 20 + 30 = 60$$

That's it! Just fancy notation for "add them up."

### Softmax Example With Cake Scores

Let's use our cake example with exponentials:

**Raw contribution scores:** [10, 5, 2]

**Step 1: Apply exponential (make positive, amplify differences)**
```math
\begin{align}
e^{10} &= 22,026 \\
e^{5} &= 148 \\
e^{2} &= 7.4
\end{align}
```

**Step 2: Sum**
$$\text{sum} = 22,026 + 148 + 7.4 = 22,181.4$$

**Step 3: Normalize (divide by sum)**
```math
\begin{align}
\text{Alice} &= 22,026 / 22,181.4 = 99.3\% \\
\text{Bob} &= 148 / 22,181.4 = 0.67\% \\
\text{Charlie} &= 7.4 / 22,181.4 = 0.03\%
\end{align}
```

**Notice what happened!** The differences got amplified:
- Before: Alice helped 2× more than Bob (10 vs 5)
- After softmax: Alice gets 148× more cake than Bob! (99.3% vs 0.67%)

**This is why softmax is perfect for attention:** It amplifies differences, making the model focus strongly on the most relevant words while still considering others slightly!

### Softmax Intuition

**Softmax is like a "fair but harsh" judge:**

- If you're the best by a little, you get rewarded A LOT
- If you're mediocre, you get very little
- If you're the worst, you get almost nothing (but never exactly zero!)

**In our dating app:**
- Alex (score 220) gets 85% of your attention
- Casey (score 180) gets 13% of your attention
- Blake (score 140) gets 2% of your attention

The differences are amplified, but everyone still gets at least a tiny bit of attention!

### Applying Softmax to Attention Scores

**Step 1: Calculate exponentials**

**What is $e$?** Euler's number, $e \approx 2.71828...$, is a special mathematical constant. Like π (pi), it appears everywhere in nature and math!

**What does $e^x$ do?**
- $e^0 = 1$ (anything to power 0 is 1)
- $e^1 = 2.718$
- $e^2 = 7.389$
- $e^{-1} = 0.368$ (negative exponents make small numbers)
- $e^{-10} = 0.000045$ (very negative → very small)

**Key properties for softmax:**
- Always positive: $e^x > 0$ for any x (even negative x!)
- Preserves ordering: if $a > b$, then $e^a > e^b$
- Amplifies differences: small difference in input → bigger difference in output

Example: 
- Scores: [1.0, 1.1, 1.2]
- Exponentials: [2.718, 3.004, 3.320] — differences amplified!

Now our calculations:
```math
\begin{align}
e^{-0.0246} &= 0.9757 \\
e^{-0.0142} &= 0.9859 \\
e^{0.1640} &= 1.1782
\end{align}
```

**How to compute $e^x$ by hand:** Use a calculator or scientific table. For understanding: remember $e^x$ grows exponentially!
- Small positive x → slightly bigger than 1
- Large positive x → very large number
- Negative x → small positive number (between 0 and 1)

**Step 2: Sum**
$$\text{sum} = 0.9757 + 0.9859 + 1.1782 = 3.1398$$

**Step 3: Normalize**
```math
\begin{align}
P(I \to I) &= 0.9757 / 3.1398 = 0.3107 \quad (31.07\%) \\
P(I \to \text{love}) &= 0.9859 / 3.1398 = 0.3140 \quad (31.40\%) \\
P(I \to \text{pizza}) &= 1.1782 / 3.1398 = 0.3753 \quad (37.53\%)
\end{align}
```

**Insight:** The word "I" pays most attention to "pizza" (37.53%)! This makes sense—the subject needs to understand what it's connected to later in the sentence.

**⚠️ Important reminder:** These attention scores (31%, 31%, 37%) are based on our **random initialization weights**! They're purely artifacts of the random numbers we chose for this example. 

**After training** on billions of examples, the learned weights would produce meaningful attention patterns—like a subject word ("I") attending strongly to its verb ("love") and object ("pizza"), or a pronoun ("it") attending to its referent ("cat"). But our random weights here are just for demonstrating the calculation process, not showing real learned behavior!

### Weighted Sum of Values

Now we create the output by mixing Value vectors according to attention weights:

$$
\begin{align}
\text{Output}_I &= 0.3107 \times \mathbf{V}_I + 0.3140 \times \mathbf{V}_{\text{love}} + 0.3753 \times \mathbf{V}_{\text{pizza}} \\
&= 0.3107 \times [-0.127, 0.211, 0.172] \\
&\quad + 0.3140 \times [-0.091, 0.381, -0.055] \\
&\quad + 0.3753 \times [0.636, 0.036, 0.277] \\
&= [-0.0395, 0.0656, 0.0534] \\
&\quad + [-0.0286, 0.1196, -0.0173] \\
&\quad + [0.2387, 0.0135, 0.1040] \\
&= [0.1706, 0.1987, 0.1401]
\end{align}
$$

This is the **Head 1 output for "I"**—a 3D vector capturing what "I" learned from attending to all words!

### Complete Head 2 (Quick Version)

Head 2 works identically but uses dimensions [3, 4, 5] and different weight matrices. To keep the chapter from turning into a giant slab of arithmetic, I'll summarize the final result here. If you want, you can treat Head 2 as a perfect calculator exercise using the same exact method:

$$\text{Head 2 output for "I"} = [0.245, -0.089, 0.156]$$

### Concatenation

Stick both head outputs together:

$$
\text{Concatenated output for "I"} = [0.1706, 0.1987, 0.1401, 0.245, -0.089, 0.156]
$$

Back to 6 dimensions, but now with multi-perspective understanding!

### Output Projection

**Why do we need this?** The concatenated heads are just "stuck together"—they haven't interacted yet. The output projection learns how to blend insights from different heads.

Weight matrix $\mathbf{W}^O$ (6×6, learned):
```math
\mathbf{W}^O = \begin{bmatrix}
0.1 & 0.2 & -0.1 & 0.3 & 0.1 & -0.2 \\
0.2 & -0.1 & 0.3 & 0.1 & 0.2 & 0.1 \\
-0.1 & 0.3 & 0.1 & -0.2 & 0.3 & 0.2 \\
0.3 & 0.1 & 0.2 & -0.1 & 0.1 & 0.3 \\
0.1 & -0.2 & 0.3 & 0.2 & -0.1 & 0.1 \\
-0.2 & 0.3 & 0.1 & 0.3 & 0.2 & -0.1
\end{bmatrix}
```

Final attention output (simplified result):
$$\text{Attention output for "I"} = [0.25, -0.31, 0.42, 0.18, -0.09, 0.33]$$

**Complete attention formula:**
```math
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}
```

This elegant formula captures everything: compute similarity scores (Q·K^T), scale them down (÷√d_k), convert to probabilities (softmax), and create a weighted combination of values (multiply by V).

Think of multi-head attention like having multiple perspectives on a scene. Head 1 might focus on "who's doing what"—it learns to connect subjects with their verbs. Head 2 might focus on "what type of thing"—it learns to group nouns by category. Head 3 (in a bigger model) might focus on emotional tone, Head 4 on temporal relationships, and so on.

The output projection is like a film director with footage from multiple cameras. Camera 1 captured the action, Camera 2 got close-ups of faces, Camera 3 filmed the background. The director doesn't just show all three side-by-side—they intelligently blend them into one coherent scene, taking the best parts of each. That's what the output projection layer learns to do with the different attention heads.

---

