## Math Symbols Quick Reference (Your Decoder Ring!)

**Before we start, let's demystify every symbol you'll see!** Anytime you encounter a symbol and think "What the hell is this?", come back to this section. Consider it your decoder ring for the tutorial!

### Common Math Symbols

| Symbol | Name | Meaning | Example |
|--------|------|---------|---------|
| = | Equals | Exactly the same | 2 + 2 = 4 |
| ≠ | Not equal | Different values | 3 ≠ 5 ("3 is not equal to 5") |
| ≈ | Approximately | Close to, not exact | π ≈ 3.14 ("pi is approximately 3.14") |
| < | Less than | Smaller than | 3 < 5 |
| > | Greater than | Larger than | 7 > 2 |
| ≤ | Less/equal | Smaller or same | 3 ≤ 3 ✓ ("3 is less than or equal to 3") |
| ≥ | Greater/equal | Larger or same | 5 ≥ 5 ✓ |
| ∞ | Infinity | Endlessly large, forever | "Count to infinity" = never finish! |
| -∞ | Negative infinity | Endlessly small/negative | We use this to "block" attention |
| × or · | Multiply | Times | 3 × 4 = 12 |
| ÷ or / | Divide | Split into parts | 12 / 4 = 3 |
| ± | Plus/minus | Could be either | ± means could be +3 or -3 |

### Greek Letters (Variables Have Fancy Names!)

Greek letters are just variable names - like using "x" but fancier!

| Symbol | Name | Commonly Used For | Example in Tutorial |
|--------|------|-------------------|---------------------|
| α | Alpha | Learning rate variations | (less common in our tutorial) |
| β | Beta | Momentum, shift parameter | β₁ = 0.9 (Adam optimizer) |
| γ | Gamma | Scale parameter | γ in LayerNorm (learned scale) |
| ε | Epsilon | Tiny safety number | ε = 0.00001 (prevents divide-by-zero) |
| η | Eta | Learning rate | η = 0.0001 (step size) |
| θ | Theta | Angle | sin(θ) means "sine of angle theta" |
| λ | Lambda | Regularization weight | Weight decay strength |
| μ | Mu | Mean (average) | μ = average of [1,2,3] = 2 |
| σ | Sigma (lowercase) | Standard deviation | How spread out numbers are |
| π | Pi | 3.14159... | The circle constant! |

**Why Greek letters?** Just a math convention! Instead of writing "learning_rate", mathematicians write η. It's shorter to write on paper!

### Math Operations Symbols

| Symbol | Name | What It Does | Example |
|--------|------|--------------|---------|
| $\sum$ | Sigma (Sum) | Add up all items | $\sum_{i=1}^{3} x_i = x_1 + x_2 + x_3$ |
| $\prod$ | Pi (Product) | Multiply all items | $\prod_{i=1}^{3} x_i = x_1 \times x_2 \times x_3$ |
| $\sqrt{x}$ | Square root | What number squared = x? | $\sqrt{9} = 3$ because $3^2 = 9$ |
| $\sqrt[3]{x}$ | Cube root | What number cubed = x? | $\sqrt[3]{8} = 2$ because $2^3 = 8$ |
| $x^2$ | Squared | Multiply by itself | $5^2 = 5 \times 5 = 25$ |
| $x^{1/2}$ | Power 1/2 | Same as square root! | $9^{1/2} = \sqrt{9} = 3$ |
| $x^{1/3}$ | Power 1/3 | Same as cube root! | $8^{1/3} = \sqrt[3]{8} = 2$ |
| $e^x$ | Exponential | 2.718... to the power x | $e^2 ≈ 7.39$ |
| $\log(x)$ | Logarithm | Inverse of exponential | If $e^2 = 7.39$, then $\log(7.39) = 2$ |
| $\sin(x)$ | Sine | Trig function (wave) | $\sin(0) = 0$ |
| $\cos(x)$ | Cosine | Trig function (wave) | $\cos(0) = 1$ |

### Calculus Symbols (Don't Panic!)

| Symbol | Name | What It Means | Think Of It As |
|--------|------|---------------|----------------|
| $\frac{\partial L}{\partial W}$ | Partial derivative | "How much does L change when I change W?" | "If I nudge weight W up, does loss go up or down?" |
| $\nabla L$ | Nabla/Gradient | All partial derivatives together | "Which direction makes loss go down?" |
| $\frac{dy}{dx}$ | Derivative | Rate of change | "How fast does y change when x changes?" |

**Simple explanation:** Derivatives tell you "if I change this input a tiny bit, how much does the output change?" It's like asking "If I turn the steering wheel 1 degree, how much does the car turn?"

### Special Notations

| Notation | Meaning | Example |
|----------|---------|---------|
| $\mathbf{x}$ | Bold letter | Vector or matrix (list of numbers) | $\mathbf{x} = [1, 2, 3]$ |
| $x_i$ | Subscript | The i-th element | If $\mathbf{x} = [5, 7, 9]$, then $x_1 = 5$, $x_2 = 7$ |
| $x^T$ | Superscript T | Transpose (flip rows↔columns) | $[1,2,3]^T$ becomes vertical |
| $\sim$ | Tilde | "Sampled from" or "drawn from" | $x \sim \mathcal{N}(0,1)$ = "x is randomly picked from a bell curve" |
| $\in$ | Element of | "Is one of" | $5 \in [1,3,5,7]$ means "5 is in this list" |
| $\odot$ | Circled dot | Element-wise multiply | $[1,2] \odot [3,4] = [1×3, 2×4] = [3,8]$ |
| $\mathcal{N}$ | Curly N | Normal distribution | Bell curve (Gaussian) |
| $\|\|x\|\|$ | Double bars | Length/norm of vector | $\|\|[3,4]\|\| = 5$ (Pythagorean theorem!) |
| $\begin{bmatrix}...\end{bmatrix}$ | Brackets | Matrix (table of numbers) | $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ |

### Function Notation

| Notation | Meaning | Example |
|----------|---------|---------|
| $f(x)$ | Function f applied to x | If $f(x) = 2x$, then $f(5) = 10$ |
| $\text{max}(a, b)$ | Maximum | Biggest of the two | $\text{max}(3, 7) = 7$ |
| $\text{argmax}(x)$ | Argument of maximum | WHICH position has the max? | $\text{argmax}([2,5,3]) = 2$ (position 2 has the max value 5) |
| $\text{softmax}(x)$ | Softmax function | Convert to probabilities (sum=1) | Explained in Chapter 5! |
| $\text{ReLU}(x)$ | ReLU activation | $\text{max}(0, x)$ | "Keep if positive, zero if negative" |

### Subscripts and Superscripts

**Subscripts** (small letters below): Usually mean index/position
- $W_1$ = "Weight matrix number 1"
- $x_i$ = "The i-th element of x"
- $\text{pos}$ = "position" (word used as subscript)

**Superscripts** (small letters above): Can mean different things!
- $x^2$ = "x squared" (power)
- $x^T$ = "x transposed" (operation)
- $W^Q$ = "Weight matrix for Query" (label, not power!)

**Context tells you which!** If it's a number (like $x^2$), it's a power. If it's a letter (like $W^Q$), it's a label!

**Pro tip:** Whenever you see a symbol you don't recognize, Ctrl+F search this section! We've explained every symbol you'll encounter in this tutorial.

---

### What You'll Learn in This Tutorial

We'll build a complete transformer from scratch, small enough to compute by hand but real enough to understand ChatGPT. You'll learn:

1. **The Input Pipeline**: How text becomes numbers computers can process
2. **The Core Architecture**: Attention, feed-forward networks, normalization
3. **The Training Process**: How random weights become intelligent through gradient descent
4. **The Generation Process**: How trained models predict and create new text
5. **The Engineering Details**: All the tricks that make it work in practice

By the end, you'll understand the core operations inside a GPT-style transformer. Production systems add scale, engineering, and extra training stages, but the foundation is the same.

---

