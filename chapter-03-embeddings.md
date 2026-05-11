## Chapter 3: Embeddings (Giving Numbers Meaning)

### The Problem: One Number Isn't Enough

We now have our sentence as token IDs: `[123, 567, 999]` representing "I love pizza"

But wait—there's a **huge problem**! Each token is just a single number, an ID like a student ID card. Think about it:
- Token 123 = "I"
- Token 567 = "love"
- Token 999 = "pizza"

These numbers tell us WHICH word it is, but they don't tell us ANYTHING about what the word MEANS. It's like having a library where books are numbered 1, 2, 3, but the numbers don't tell you if a book is about cooking, science, or history!

**The fundamental question:** How do we represent the MEANING of a word using numbers that a computer can process?

### Describing Things with Multiple Properties

Let's think about how you'd describe "pizza" to someone who's never seen it:
- "It's **food**" (not a person, not a tool)
- "It's **circular**" (has a shape)
- "It's **tasty**" (positive quality)
- "It's **Italian**" (has cultural origin)
- "It's a **noun**" (grammar category)
- "It's **solid**" (not liquid, not abstract)

Notice how you need MULTIPLE pieces of information to fully describe pizza? One single fact isn't enough!

**Another example—describing yourself:**
- Height: 5.7 feet
- Age: 25 years
- Happiness level: 8/10
- Energy level: 6/10
- Creativity: 7/10
- Athleticism: 4/10

Each of these is a different **property** or **feature** that describes you. Together, they paint a complete picture!

### From One Number to Many: Introducing Dimensions

This is exactly what we need for words! Instead of representing each word with ONE number (the token ID), we'll represent it with MULTIPLE numbers—each number capturing a different property or feature of the word's meaning.

**The key idea:** We'll give each token multiple "slots" to store different aspects of its meaning. We call each slot a **dimension**.

**Simple example with 3 dimensions:**

Imagine we have 3 dimensions, and after training, they learn to represent:
- Dimension 1: "How positive is this word?" (scale from -1 to +1)
- Dimension 2: "Is it food-related?" (scale from -1 to +1)
- Dimension 3: "Is it a noun or verb?" (scale from -1 to +1)

Then our words might look like:
```
"pizza":  [0.2,   0.9,   0.8]
          ↑      ↑      ↑
       slightly food!  noun!
       positive

"love":   [0.9,  -0.3,  -0.7]
          ↑      ↑       ↑
       very   not food  verb!
       positive

"hate":   [-0.8, -0.2,  -0.6]
           ↑      ↑       ↑
        negative not food verb!
```

Do you see the pattern? 
- "love" (0.9) and "hate" (-0.8) are OPPOSITE on dimension 1—perfect! They have opposite sentiment.
- "pizza" (0.9) is HIGH on dimension 2—yes, it's food!
- "love" and "hate" are both negative on dimension 3—they're both verbs!

**The magic:** These multiple numbers let the model understand that "love" and "hate" are similar (both verbs, both emotions) yet opposite (positive vs negative)!

### More Dimensions = Richer Meaning

In our tutorial, we'll use **6 dimensions** for each word. In real systems like ChatGPT, they use **12,288 dimensions**!

**Why so many?** Because language is incredibly rich and nuanced! Think about all the properties a word could have:
- Is it positive or negative? (sentiment)
- Is it concrete or abstract? (concreteness)
- Is it a noun, verb, adjective? (grammar)
- Is it formal or casual? (register)
- Is it about people, objects, or ideas? (category)
- Is it common or rare? (frequency)
- Is it modern or old-fashioned? (temporal)
- Does it relate to sight, sound, taste, touch, smell? (sensory)
- Is it technical or everyday? (domain)
- Is it literal or metaphorical? (usage)

With only 6 dimensions, we can capture maybe 6 major properties. With 12,288 dimensions, we can capture incredibly subtle nuances of meaning!

**Important reality check:** We WON'T explicitly tell the model "dimension 1 = sentiment, dimension 2 = food-ness." The model will **learn** what each dimension should represent during training, automatically discovering which properties are most useful for predicting the next word!

After training:
- We CAN'T look at dimension 3 and say "oh, this is the noun/verb dimension"
- The dimensions DON'T have labels
- But the model DOES use them effectively to capture meaning
- Researchers can analyze patterns and guess what dimensions learned, but it's fuzzy

Think of it like how your brain works—you can tell "dog" and "puppy" are similar, but you can't point to specific neurons and say "neuron #4829 stores dog-ness." The knowledge is distributed across many neurons working together. Same here!

### Quick Math Refresher: Vectors and Matrices

Before we continue, let's learn the mathematical tools we need:

**Vector:** A list of numbers representing multiple properties
- Row vector: $[1, 2, 3]$ — this could be 3 properties of something
- Column vector : 

$$
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
$$


**Why called "vector"?** In math/physics, a vector is something with multiple components (like velocity has speed + direction). Our word vectors have multiple meaning components!

**Example:** The word "pizza" with 6 dimensions:
$$\text{pizza} = [0.90, -0.10, -0.20, -0.80, 0.50, 0.75]$$

This vector has 6 numbers, capturing 6 different learned properties of "pizza."

**Matrix:** A table of vectors stacked together
```math
\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}
```
This is a 2×3 matrix: 2 rows (could be 2 words), 3 columns (could be 3 dimensions per word)

**Reading dimensions:** This matrix has:
- Shape: 2 rows × 3 columns (we write rows first, then columns)
- Row 1: $[1, 2, 3]$ — could be properties of word #1
- Row 2: $[4, 5, 6]$ — could be properties of word #2

**Matrix multiplication:** This is how we transform data! We'll use this extensively in the transformer.

To multiply matrix $\mathbf{A}$ (2×3) by matrix $\mathbf{B}$ (3×2):

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}, \quad
\mathbf{B} = \begin{bmatrix}
7 & 8 \\
9 & 10 \\
11 & 12
\end{bmatrix}
$$

Result is 2×2 matrix (outer dimensions):

$$
\mathbf{C} = \mathbf{A} \times \mathbf{B} = \begin{bmatrix}
c_{11} & c_{12} \\
c_{21} & c_{22}
\end{bmatrix}
$$

**How to calculate each element:**

$c_{11}$ (row 1 of A, column 1 of B):
$$c_{11} = (1)(7) + (2)(9) + (3)(11) = 7 + 18 + 33 = 58$$

$c_{12}$ (row 1 of A, column 2 of B):
$$c_{12} = (1)(8) + (2)(10) + (3)(12) = 8 + 20 + 36 = 64$$

$c_{21}$ (row 2 of A, column 1 of B):
$$c_{21} = (4)(7) + (5)(9) + (6)(11) = 28 + 45 + 66 = 139$$

$c_{22}$ (row 2 of A, column 2 of B):
$$c_{22} = (4)(8) + (5)(10) + (6)(12) = 32 + 50 + 72 = 154$$

**Final result:**
```math
\mathbf{C} = \begin{bmatrix}
58 & 64 \\
139 & 154
\end{bmatrix}
```

**The pattern:** To get element at position (row i, column j), take the i-th row of the first matrix and j-th column of the second matrix, multiply corresponding elements, and sum them up.

**Important rule:** To multiply A×B, the number of columns in A must equal the number of rows in B!
- (2×3) × (3×2) ✓ Works! (middle numbers match: 3 = 3)
- (2×3) × (2×4) ✗ Doesn't work! (3 ≠ 2)

Now let's use these tools to build our embedding system!

### The Embedding Table: A Giant Lookup Dictionary

Remember: We have 50,000 tokens in our vocabulary, and we want to represent each with 6 dimensions.

**Key terminology:**
- **Token ID:** The single number identifying a word (e.g., 999 for "pizza")
- **Embedding:** The vector of 6 numbers representing the word's meaning
- **Dimension:** Each individual number in the embedding vector (we have 6 dimensions)
- **$d_{\text{model}}$:** The fancy name for "how many dimensions we use" (in our case, 6)

**Why "$d_{\text{model}}$"?** Think of it as "dimension of the model" — it's the size of the vectors flowing through our entire transformer. Every word will be represented by $d_{\text{model}}$ numbers throughout the entire model.

**Our choice:** $d_{\text{model}} = 6$ (small, so you can calculate by hand!)
**GPT-3's choice:** $d_{\text{model}} = 12,288$ (huge, captures incredibly subtle meanings!)

### Creating the Embedding Lookup Table

We create a giant table (matrix) with dimensions: `[vocab_size × d_model]` = `[50,000 × 6]`

Think of it like a spreadsheet:
- **50,000 rows:** One row for each word in our vocabulary
- **6 columns:** One column for each dimension/property

```
Embedding Matrix E (showing just 3 words out of 50,000):

Token ID  Word      dim₀    dim₁    dim₂    dim₃    dim₄    dim₅
--------  ----      ----    ----    ----    ----    ----    ----
   123    "I"       0.01   -0.20    0.30    0.15   -0.05    0.11
   567    "love"   -0.40    0.60    0.00    0.25    0.90   -0.30
   999    "pizza"   0.90   -0.10   -0.20   -0.80    0.50    0.75
   ...    ...       ...     ...     ...     ...     ...     ...
 49999    (last)    0.23    0.45   -0.67    0.12    0.89   -0.34
```

**How to read this:**
- Row 123 (token "I") has embedding: `[0.01, -0.20, 0.30, 0.15, -0.05, 0.11]`
- Row 999 (token "pizza") has embedding: `[0.90, -0.10, -0.20, -0.80, 0.50, 0.75]`

**Where do these numbers come from initially?** They start **completely random**! Small random numbers near zero (we'll explain the specific initialization strategy later). The model has NO IDEA what any word means at the start.

**How do they become meaningful?** Through training! The model will adjust these numbers billions of times, gradually learning to encode meaningful properties. After training:
- Similar words end up with similar vectors (close in 6-dimensional space)
- Opposite words end up far apart or pointing in opposite directions
- The dimensions self-organize to capture useful patterns

### Looking Up Embeddings: From Token ID to Meaning Vector

When we want the embedding for token 123 ("I"), we simply grab row 123 from the table:

$$\text{embedding}_{123} = \mathbf{E}[123, :] = [0.01, -0.20, 0.30, 0.15, -0.05, 0.11]$$

**What does the notation mean?**
- $\mathbf{E}$ = the embedding matrix (our giant table)
- $[123, :]$ = "row 123, all columns"
- Result: A 6-dimensional vector representing "I"

**For our entire sentence "I love pizza":**

Token IDs: `[123, 567, 999]`

Look up each:
- Token 123 → `[0.01, -0.20, 0.30, 0.15, -0.05, 0.11]`
- Token 567 → `[-0.40, 0.60, 0.00, 0.25, 0.90, -0.30]`
- Token 999 → `[0.90, -0.10, -0.20, -0.80, 0.50, 0.75]`

Stack them into a matrix:

$$
\mathbf{X} = \begin{bmatrix}
0.01 & -0.20 & 0.30 & 0.15 & -0.05 & 0.11 \\
-0.40 & 0.60 & 0.00 & 0.25 & 0.90 & -0.30 \\
0.90 & -0.10 & -0.20 & -0.80 & 0.50 & 0.75
\end{bmatrix}
$$

**Shape:** $3 \times 6$ (3 words, 6 dimensions each)

**How to read this matrix:**
- Row 1 = embedding for "I"
- Row 2 = embedding for "love"  
- Row 3 = embedding for "pizza"
- Column 1 (dim₀) = first learned property (all 3 words have values for this)
- Column 2 (dim₁) = second learned property
- ... and so on

This matrix is what flows into our transformer! Each row represents one word's meaning as a 6-dimensional vector.

### What Each Dimension Learns (After Training)

Remember: we **don't** tell the model what each dimension should mean. But after training on billions of sentences, patterns emerge! Here's what might happen (this is speculative—the model figures it out):

**⚠️ IMPORTANT DISCLAIMER - Read This First!**

The following examples are **thought experiments to help you build intuition**. In reality, concepts are NOT stored neatly in single dimensions! 

**The truth:** A concept like "concreteness" would be represented as a complex pattern distributed across MANY dimensions simultaneously, not in one tidy slot. Real embeddings are much more complex and largely uninterpretable to humans.

**We use these simplified labels purely as teaching tools** to help you understand that embeddings capture meaningful properties. Don't expect to open a trained model and find a clean "sentiment dimension"!

**With that clarified, here are hypothetical examples:**

**Dimension 0:** Could learn something like "concreteness" (thought experiment!)
- "pizza" = 0.90 (very concrete, tangible object)
- "love" = -0.40 (abstract concept)
- "I" = 0.01 (neutral—a pronoun)

**Dimension 1:** Could learn "sentiment/emotion"
- "love" = 0.60 (positive emotion!)
- "pizza" = -0.10 (neutral object, slightly negative because not all foods are loved?)
- "I" = -0.20 (neutral pronoun)

**Dimension 2:** Could learn "commonly used in positive contexts"
- "I" = 0.30 (commonly starts positive sentences)
- "love" = 0.00 (appears in both positive and negative contexts)
- "pizza" = -0.20 (food, neutral)

**Dimension 3:** Could learn "relates to physical objects vs actions"
- "I" = 0.15 (person-related)
- "love" = 0.25 (action/feeling)
- "pizza" = -0.80 (physical object!)

**And so on...**

The model discovers: "To predict what word comes next, it helps to know if the current word is concrete or abstract, positive or negative, a noun or verb, etc." So it learns to encode exactly those properties that are useful!

**The beautiful emergence:** After training, if you look at embeddings for similar words, they'll be close together:
- "pizza" = `[0.90, -0.10, -0.20, -0.80, 0.50, 0.75]`
- "pasta" ≈ `[0.88, -0.12, -0.18, -0.79, 0.48, 0.73]` ← Very similar!
- "love" = `[-0.40, 0.60, 0.00, 0.25, 0.90, -0.30]`
- "adore" ≈ `[-0.38, 0.63, 0.02, 0.27, 0.88, -0.28]` ← Very similar!

The numbers naturally cluster for similar meanings because similar words appear in similar contexts, and the model learns from those patterns!

### Why More Dimensions?

**With 2 dimensions:** We could maybe capture "positive vs negative" and "noun vs verb"
- Very limited understanding
- Can't distinguish between "happy" and "ecstatic" (both positive, both adjectives)

**With 6 dimensions** (our tutorial): We can capture several major properties
- Good enough to see basic patterns
- Small enough to calculate by hand!

**With 12,288 dimensions** (GPT-3): We can capture incredibly subtle nuances
- Difference between "happy" and "joyful" and "content" and "ecstatic"
- Context-dependent meanings (bank = financial institution vs river bank)
- Metaphorical uses, sarcasm indicators, formality levels, cultural connotations
- Everything that makes language rich and complex!

**Analogy:** Describing a painting
- 2 colors: Can barely sketch the idea
- 6 colors: Can capture the basic scene
- 12,288 colors: Can reproduce every subtle shade and texture

Same with word meanings! More dimensions = richer, more nuanced understanding.

### How Do Random Numbers Become Meaningful? (A Preview)

**You might be wondering:** "If embeddings start random, how do they learn to capture meaning?"

**The short answer:** Through training! The model makes predictions, gets feedback on mistakes, and adjusts the numbers to improve. Over billions of examples, meaningful patterns emerge.

**The slightly longer answer:** Imagine you're learning to bake. First attempt, your cake is terrible (random guesses). Someone tastes it and says "too sweet, needs more flour." You adjust. Next attempt is better. After hundreds of cakes, you've perfected the recipe. The embedding numbers work the same way—they get adjusted based on feedback until they're "perfect" for predicting the next word.

**Don't worry about the details yet!** We'll dive deep into exactly HOW this learning process works in **Chapter 11: Training the Transformer**. For now, just understand:
- Embeddings START random (meaningless numbers)
- Training ADJUSTS them based on prediction errors  
- After training, they CAPTURE meaning (similar words = similar vectors)

This is the magic of machine learning—meaningful structure emerges from random initialization through the training process!

**For this chapter, remember:**
- Looking up an embedding is super fast—just grab the row from the table
- Each word is represented by $d_{\text{model}}$ numbers (6 in our case)
- These numbers capture different properties/features of the word's meaning
- The model will learn what those properties should be during training

---

