## Building a Transformer: The Complete Guide from Paper to Production

> *A Super-Friendly Tutorial for Curious Learners, Students, and Practitioners*

**Author:** [Rimom Costa](mailto:rimomcosta@gmail.com)

---

## What Makes This Tutorial Special?

This is **not** your typical AI tutorial. This is a complete, from-first-principles guide to understanding transformers—the architecture powering ChatGPT, Claude, DALL-E, and virtually every modern AI breakthrough.

**What sets this apart:**

- **Hand-calculable examples** - Uses tiny numbers (6 dimensions instead of 12,288) so you can verify EVERY calculation by hand
- **Accessible without being shallow** - Explains concepts clearly enough for beginners while staying useful for serious technical readers
- **Complete coverage** - From tokenization to training to production ChatGPT (including RLHF)
- **No magic boxes** - Every formula explained, every design choice justified
- **Visual learning** - Analogies, examples, and intuitive explanations throughout
- **Career-focused** - Teaches what you'll ACTUALLY do in industry (hint: not training from scratch!)

**After this course, you'll understand:**
- How ChatGPT actually works under the hood
- Why transformers revolutionized AI
- The complete training pipeline (pre-training → fine-tuning → RLHF)
- How to implement transformers from scratch
- What makes GPT different from BERT and T5

---

## Course Contents

**Click on any chapter to jump directly to that section in the course!**

### Preparation
- [Introduction: What Are Transformers Really?](00-introduction.md#introduction-what-are-transformers-really) - The big-picture starting point

### Foundation (Chapters 0-4)
- [**Chapter 0:** The Grand Vision](chapter-00-grand-vision.md#chapter-0-the-grand-vision) - What problem are we solving?
- [**Chapter 1:** Building Our Vocabulary](chapter-01-building-our-vocabulary.md#chapter-1-building-our-vocabulary-the-dictionary) - The token dictionary
- [**Chapter 2:** Tokenization](chapter-02-tokenization.md#chapter-2-tokenization-chopping-text-into-pieces) - Byte-Pair Encoding (BPE) explained
- [**Chapter 3:** Embeddings](chapter-03-embeddings.md#chapter-3-embeddings-giving-numbers-meaning) - Giving numbers meaning
- [**Chapter 4:** Positional Encoding](chapter-04-positional-encoding.md#chapter-4-positional-encoding-teaching-word-order) - Teaching word order with sine waves
  - [Part 1: Understanding the Core Problem](chapter-04-positional-encoding.md#part-1-understanding-the-core-problem)
  - [Part 2: Failed Attempts (Learning from Mistakes)](chapter-04-positional-encoding.md#part-2-failed-attempts-learning-from-mistakes)
  - [Part 3: The Breakthrough — Understanding Waves](chapter-04-positional-encoding.md#part-3-the-breakthrough-—-understanding-waves)
  - [Part 4: Building the Solution Step-by-Step](chapter-04-positional-encoding.md#part-4-building-the-solution-stepbystep)
  - [Part 5: Calculating Position Encodings (Hands-On!)](chapter-04-positional-encoding.md#part-5-calculating-position-encodings-handson)
  - [Part 6: Combining Position with Word Meaning](chapter-04-positional-encoding.md#part-6-combining-position-with-word-meaning)
  - [Part 7: Critical Question About Training](chapter-04-positional-encoding.md#part-7-critical-question-about-training)
  - [Part 8: Why This Solution Is Beautiful](chapter-04-positional-encoding.md#part-8-why-this-solution-is-beautiful)
  - [Part 9: Summary and Key Takeaways](chapter-04-positional-encoding.md#part-9-summary-and-key-takeaways)

### Core Architecture (Chapters 5-10)
- [**Chapter 5:** Multi-Head Self-Attention](chapter-05-multi-head-self-attention.md#chapter-5-multihead-selfattention-the-heart) - The heart of transformers
  - [Part 1: Understanding the Core Concept](chapter-05-multi-head-self-attention.md#part-1-understanding-the-core-concept)
  - [Part 2: Multi-Head Attention (Why Multiple Perspectives?)](chapter-05-multi-head-self-attention.md#part-2-multihead-attention-why-multiple-perspectives)
  - [Part 3: The Mathematics (Step-by-Step Calculations)](chapter-05-multi-head-self-attention.md#part-3-the-mathematics-stepbystep-calculations)
  - [Part 4: Computing Similarity (The Dot Product)](chapter-05-multi-head-self-attention.md#part-4-computing-similarity-the-dot-product)
  - [Part 5: Converting Scores to Probabilities (Softmax)](chapter-05-multi-head-self-attention.md#part-5-converting-scores-to-probabilities-softmax)
- [**Chapter 6:** Dropout](chapter-06-dropout.md#chapter-6-dropout-the-training-safety-net) - The training safety net
- [**Chapter 7:** Feed-Forward Network](chapter-07-feed-forward-network.md#chapter-7-feedforward-network-individual-processing) - Individual word processing
- [**Chapter 8:** Residual Connections & Layer Normalization](chapter-08-residual-connections-layer-normalization.md#chapter-8-residual-connections--layer-normalization) - Gradient highways
- [**Chapter 9:** Stacking Transformer Blocks](chapter-09-stacking-transformer-blocks.md#chapter-9-stacking-transformer-blocks) - Building depth
- [**Chapter 10:** The Output Head](chapter-10-output-head.md#chapter-10-the-output-head-predicting-next-token) - Predicting the next token

### Training & Inference (Chapters 11-13)
- [**Chapter 11:** Training the Transformer](chapter-11-training-the-transformer.md#chapter-11-training-the-transformer-the-learning-process) - Backpropagation, loss functions, optimizers
- [**Chapter 12:** Causal Masking](chapter-12-causal-masking.md#chapter-12-causal-masking-no-cheating) - Preventing the model from cheating
- [**Chapter 13:** Inference](chapter-13-inference.md#chapter-13-inference-using-the-trained-model) - Using the trained model (with KV cache optimization!)

### Advanced Topics (Chapters 14-17)
- [**Chapter 14:** All the Hyperparameters](chapter-14-all-the-hyperparameters.md#chapter-14-all-the-hyperparameters-the-control-panel) - The complete control panel
- [**Chapter 15:** Additional Techniques](chapter-15-additional-techniques.md#chapter-15-additional-techniques) - Gradient accumulation, mixed precision, checkpointing
- [**Chapter 16:** Common Training Problems & Solutions](chapter-16-common-training-problems-solutions.md#chapter-16-common-training-problems--solutions) - Debugging guide
- [**Chapter 17:** Putting It All Together](chapter-17-putting-it-all-together.md#chapter-17-putting-it-all-together-complete-example) - Complete end-to-end example

### Real-World Applications (Chapters 18-21)
- [**Chapter 18:** From Language Model to ChatGPT](chapter-18-from-language-model-to-chatgpt.md#chapter-18-from-language-model-to-chatgpt-the-three-training-stages) - The three training stages (pre-training, fine-tuning, RLHF)
  - [Part 1: Pre-Training (Building the Foundation)](chapter-18-from-language-model-to-chatgpt.md#part-1-pretraining-building-the-foundation)
  - [Part 2: Fine-Tuning (Specialization)](chapter-18-from-language-model-to-chatgpt.md#part-2-finetuning-specialization)
  - [Part 3: Instruction Tuning & RLHF (Making It Helpful)](chapter-18-from-language-model-to-chatgpt.md#part-3-instruction-tuning--rlhf-making-it-helpful)
  - [Part 4: Practical Implications](chapter-18-from-language-model-to-chatgpt.md#part-4-practical-implications)
  - [Part 5: The Timeline of Modern LLMs](chapter-18-from-language-model-to-chatgpt.md#part-5-the-timeline-of-modern-llms)
  - [Part 6: Key Takeaways](chapter-18-from-language-model-to-chatgpt.md#part-6-key-takeaways)
- [**Chapter 19:** Three Transformer Architectures](chapter-19-three-transformer-architectures.md#chapter-19-the-three-transformer-architectures-decoder-encoder-encoderdecoder) - Understanding encoder vs decoder vs encoder-decoder
  - [Part 1: Understanding the Three Architectures](chapter-19-three-transformer-architectures.md#part-1-understanding-the-three-architectures)
  - [Part 2: Key Differences Explained](chapter-19-three-transformer-architectures.md#part-2-key-differences-explained)
  - [Part 3: Which One Should You Use?](chapter-19-three-transformer-architectures.md#part-3-which-one-should-you-use)
  - [Part 4: What Makes Decoder-Only Special (What You Learned)](chapter-19-three-transformer-architectures.md#part-4-what-makes-decoderonly-special-what-you-learned)
  - [Part 5: The Missing Piece (Cross-Attention in Encoder-Decoder)](chapter-19-three-transformer-architectures.md#part-5-the-missing-piece-crossattention-in-encoderdecoder)
  - [Part 6: What You Learned vs What Exists](chapter-19-three-transformer-architectures.md#part-6-what-you-learned-vs-what-exists)
  - [Part 7: Modern Landscape (What's Actually Used)](chapter-19-three-transformer-architectures.md#part-7-modern-landscape-whats-actually-used)
  - [Part 8: Summary - What You Actually Know](chapter-19-three-transformer-architectures.md#part-8-summary--what-you-actually-know)
- [**Chapter 20:** Quick Quizzes](chapter-20-quick-quizzes.md#chapter-20-quick-quizzes-test-yourself) - Test your understanding
- [**Chapter 21:** Going Further](chapter-21-going-further.md#chapter-21-going-further) - Next steps and resources

### Appendix
- [Math Symbols Quick Reference](appendix-math-symbols-quick-reference.md#math-symbols-quick-reference-your-decoder-ring) - Your decoder ring for all the notation

---

## Who Is This For?

### Perfect for:
- **Complete beginners** who want to understand AI from first principles
- **Software engineers** transitioning into ML/AI
- **Students** learning about transformers in courses
- **Researchers** who want to understand the fundamentals deeply
- **Educators** looking for teaching materials
- **Curious minds** who want to know how ChatGPT actually works

### Prerequisites:
- **Minimal:** Basic arithmetic (addition, multiplication)
- **Helpful but not required:** High school algebra, basic Python
- **Not required:** Advanced calculus, linear algebra, or ML experience

The tutorial builds everything from the ground up, explaining even the math notation!

---

## Key Learning Outcomes

After completing this tutorial, you will:

- ✅ Understand how words become numbers (embeddings)
- ✅ Grasp how transformers know word order (positional encoding)
- ✅ Master self-attention and why it's revolutionary
- ✅ Understand the complete training process (loss, gradients, backpropagation)
- ✅ Know the difference between pre-training and fine-tuning
- ✅ Understand how ChatGPT differs from base GPT-3
- ✅ Be able to implement a transformer from scratch
- ✅ Read and understand modern AI research papers
- ✅ Debug common training issues
- ✅ Know what you'll actually do in an AI/ML career

---

## Getting Started

1. **Clone this repository**
   ```bash
   git clone git@github-rimomcosta:rimomcosta/Transformers-for-absolute-dummies.git
   cd Transformers-for-absolute-dummies
   ```

2. **Read the course**
   - Start with [`00-introduction.md`](00-introduction.md), then read the chapter files in order
   - Grab paper and pencil to follow along with calculations
   - Take your time—understanding is more important than speed!

3. **Try the calculations yourself**
   - Don't just read—actually calculate the examples
   - The numbers are small enough to do by hand
   - This is where real understanding happens!

---

## What Makes This Tutorial Unique?

### 1. One Unified Explanation Style
Every concept aims to do both jobs at once:
- Build intuition with concrete analogies and plain language
- Preserve the real mathematics and implementation logic
- Avoid splitting into separate "simple" and "technical" versions

### 2. Hand-Verifiable Math
Unlike most tutorials that use production-scale dimensions:
- **Typical tutorial:** "Imagine a 768-dimensional vector..." (impossible to calculate)
- **This tutorial:** "Here's a 6-dimensional vector: [0.2, -0.1, 0.5, 0.3, -0.4, 0.1]" (you can verify every step!)

### 3. Complete Training AND Inference
Most guides only show inference (using a trained model). This tutorial covers:
- How the model learns (training)
- How it predicts (inference)
- How to optimize for production (KV cache)

### 4. Real Career Guidance
Explains what you'll ACTUALLY do in industry:
- You DON'T train frontier-scale models from scratch ($10M+ to much higher budgets, depending on the system)
- You DO fine-tune existing models ($100-$1000 budget) or use other efficient adaptation techniques when appropriate
- Understanding the difference is critical for career success!

### 5. Modern Content
- Covers decoder-only transformers (GPT-style), the most common foundation behind modern LLMs
- Explains the core training stages behind ChatGPT-style assistants
- Includes recent optimizations (KV cache, flash attention concepts)

---

## Contributing

First off, **thank you** for considering contributing to this project!

This tutorial aims to be the most accessible and comprehensive transformer guide ever created. Every contribution—whether it's fixing a typo, adding a diagram, or creating interactive examples—helps make AI education more accessible.

### Ways to Contribute

#### Visual & Design Contributions
- **Illustrations:** Create diagrams for attention mechanisms, embedding spaces, etc.
- **Infographics:** Design visual summaries of complex concepts
- **Animations:** Build animated visualizations of transformer operations
- **Web Design:** Help create a beautiful, accessible website
- **UI/UX:** Improve the reading experience

#### Code Contributions
- **PyTorch Implementation:** Complete, commented implementation matching the tutorial
- **Jupyter Notebooks:** Interactive notebooks with step-by-step execution
- **Visualization Tools:** Interactive demos of attention, embeddings, etc.
- **Web Demos:** Browser-based implementations
- **Testing Frameworks:** Tools to verify calculations

#### Content Contributions
- **Proofreading:** Fix typos, grammar, and clarity issues
- **Additional Examples:** Add more worked examples
- **Analogies:** Suggest better analogies for difficult concepts
- **Exercises:** Create practice problems with solutions
- **Quizzes:** Interactive self-assessment tools
- **Translations:** Translate to other languages

#### Documentation
- **API Documentation:** If code is added, document it thoroughly
- **Setup Guides:** Help others get started with implementations
- **Troubleshooting:** Document common issues and solutions
- **FAQ:** Add frequently asked questions

#### Testing & Feedback
- **Beta Testing:** Work through the tutorial and report issues
- **Accuracy Review:** Verify mathematical correctness
- **Pedagogical Review:** Test with diverse audiences
- **Accessibility Review:** Ensure content is accessible to all

### Contribution Guidelines

#### Before You Start

1. **Check existing issues:** Someone might already be working on it
2. **Open an issue:** Discuss major changes before investing time
3. **Read the license:** Understand the [licensing terms](#license)
4. **Keep the tone:** Maintain the friendly, accessible style

#### Contribution Process

1. **Fork the repository**
   ```bash
   git clone git@github-rimomcosta:YOUR_USERNAME/Transformers-for-absolute-dummies.git
   cd Transformers-for-absolute-dummies
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Make your changes**
   - Write clear, descriptive commit messages
   - Keep changes focused (one feature/fix per PR)
   - Test your changes thoroughly

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Explain why the change is valuable

#### Commit Message Guidelines

Use clear, descriptive commit messages:
- `Add: new section on flash attention`
- `Fix: typo in chapter 5, paragraph 3`
- `Improve: clarity of attention mechanism explanation`
- `Update: diagram for multi-head attention`
- `Docs: add setup instructions for PyTorch`

#### Code Style (for code contributions)

- **Python:** Follow PEP 8, use type hints
- **Comments:** Explain WHY, not just WHAT
- **Naming:** Clear, descriptive variable names
- **Documentation:** Docstrings for all functions/classes

Example:
```python
def calculate_attention_scores(query: np.ndarray, key: np.ndarray) -> np.ndarray:
    """
    Calculate attention scores between query and key vectors.

    Args:
        query: Query vector of shape (d_k,)
        key: Key vector of shape (d_k,)

    Returns:
        Attention score (scalar value)

    Example:
        >>> query = np.array([1.0, 0.5])
        >>> key = np.array([0.8, 0.6])
        >>> calculate_attention_scores(query, key)
        1.1
    """
    return np.dot(query, key)
```

#### Content Style (for written contributions)

##### Tone
- **Friendly and encouraging:** "Great! Now let's see..."
- **Avoid condescension:** Never "Obviously..." or "Simply..."
- **Inclusive language:** Use "we" and "our"

##### Structure
- **Short paragraphs:** 3-5 sentences max
- **Clear headings:** Descriptive and hierarchical
- **Examples first:** Show, then explain
- **Unified explanation style:** Build plain-language intuition while preserving the real technical detail

##### Formatting
- **Bold** for emphasis
- `code` for technical terms
- > Blockquotes for important notes
- Lists for clarity
- Tables for comparisons

##### Math Notation
- Define symbols before using them
- Use LaTeX for complex equations: `$E = mc^2$`
- Show numerical examples after formulas
- Explain in words what the math means

Example:
```markdown
#### Understanding the Dot Product

The **dot product** measures similarity between two vectors.

**Formula:**
$\text{score} = \vec{q} \cdot \vec{k} = q_1 k_1 + q_2 k_2 + ... + q_n k_n$

**Example:**
Query: $\vec{q} = [1.0, 0.5]$
Key: $\vec{k} = [0.8, 0.6]$

$\text{score} = (1.0 \times 0.8) + (0.5 \times 0.6) = 0.8 + 0.3 = 1.1$

**Intuition:** Higher scores mean the vectors point in similar directions!
```

### Specific Help Needed

#### High Priority

1. **Visual Diagrams**
   - Multi-head attention mechanism
   - Transformer block architecture
   - Training pipeline (pre-training → fine-tuning → RLHF)
   - Positional encoding wave patterns
   - Gradient flow through residual connections

2. **Interactive Web Version**
   - Responsive design
   - Table of contents with smooth scrolling
   - Code syntax highlighting
   - Mobile-friendly layout
   - Dark mode support

3. **PyTorch Implementation**
   - Heavily commented code matching the tutorial
   - Step-by-step execution examples
   - Debugging utilities
   - Visualization hooks

4. **Video Walkthroughs**
   - Key concept explanations
   - Hand-calculation demonstrations
   - Step-through of complete examples

#### Medium Priority

5. **Jupyter Notebooks**
   - Interactive exercises
   - Executable code cells
   - Inline visualizations

6. **Additional Examples**
   - More sentence processing examples
   - Different language examples
   - Edge cases and corner cases

7. **Exercises & Quizzes**
   - Progressive difficulty levels
   - Immediate feedback
   - Explanations for wrong answers

8. **Translations**
   - Spanish
   - Portuguese
   - Mandarin
   - Hindi
   - French
   - German

#### Future Considerations

9. **Advanced Topics**
   - Flash Attention
   - Mixture of Experts
   - Sparse Attention
   - Efficient Transformers

10. **Related Architectures**
    - Vision Transformers (ViT)
    - Diffusion Transformers
    - Multimodal transformers

### Reporting Bugs

Found an error? Please help us fix it!

#### For Content Issues
- **What:** Quote the problematic text
- **Where:** Chapter and section
- **Issue:** What's wrong (typo, factual error, clarity)
- **Suggestion:** How to fix it (if you have one)

#### For Code Issues
- **Environment:** OS, Python version, dependencies
- **Steps to reproduce:** Exact steps to trigger the bug
- **Expected behavior:** What should happen
- **Actual behavior:** What actually happens
- **Error messages:** Full error output

### Suggesting Enhancements

Have an idea? We'd love to hear it!

**Good enhancement suggestions include:**
- Clear description of the enhancement
- Explanation of why it's valuable
- Examples of how it would work
- Consideration of alternatives

### Contributor Agreement

By contributing, you agree to the terms in the [License](#license):

- Your contribution will be credited
- You retain copyright to your work
- You grant a royalty-free license to the author
- You affirm you have the right to contribute

Significant contributors will be acknowledged in the README and documentation.

### Getting Help

Stuck? Need guidance?

- **Discussions:** For questions and general discussion
- **Issues:** For specific bugs or feature requests
- **Email:** For private inquiries: [rimomcosta@gmail.com](mailto:rimomcosta@gmail.com)

### Code of Conduct

#### Our Pledge

We pledge to make participation in this project a harassment-free experience for everyone, regardless of:
- Age
- Body size
- Disability
- Ethnicity
- Gender identity and expression
- Level of experience
- Nationality
- Personal appearance
- Race
- Religion
- Sexual identity and orientation

#### Our Standards

**Positive behavior:**
- Being respectful and inclusive
- Accepting constructive criticism gracefully
- Focusing on what's best for the community
- Showing empathy toward others

**Unacceptable behavior:**
- Trolling, insulting, or derogatory comments
- Public or private harassment
- Publishing others' private information
- Other conduct inappropriate in a professional setting

#### Enforcement

Project maintainers have the right to remove, edit, or reject comments, commits, code, issues, and other contributions that don't align with this Code of Conduct.

### Recognition

Contributors will be recognized in several ways:

1. **GitHub Contributors:** Automatically listed by GitHub
2. **CONTRIBUTORS.md:** Special recognition for significant contributions
3. **In-document Attribution:** For major content additions
4. **Project Website:** Hall of fame on the website (when built)

### Project Structure

```
Transformers-for-absolute-dummies/
├── README.md        # Project overview, course index, contribution guide, and license terms
├── 00-introduction.md
├── chapter-*.md     # Individual course chapters
├── appendix-math-symbols-quick-reference.md
├── code/            # Code implementations (coming soon)
│   ├── pytorch/     # PyTorch implementation
│   ├── tensorflow/  # TensorFlow implementation
│   └── numpy/       # Pure NumPy implementation
├── notebooks/       # Jupyter notebooks (coming soon)
├── diagrams/        # Visual diagrams (coming soon)
├── website/         # Web version (coming soon)
└── translations/    # Translations (coming soon)
```

### Learning Resources for Contributors

Want to contribute but need to learn more first?

- **Git & GitHub:** [GitHub Guides](https://guides.github.com/)
- **Markdown:** [Markdown Guide](https://www.markdownguide.org/)
- **LaTeX Math:** [LaTeX Math Symbols](https://www.overleaf.com/learn/latex/List_of_Greek_letters_and_math_symbols)
- **Transformers:** Start with [`00-introduction.md`](00-introduction.md), then read the chapter files in order

### Review Process

1. **Initial review:** Within 1 week
2. **Feedback:** Clear, constructive comments
3. **Iterations:** Work together to refine
4. **Merge:** Once approved by maintainer
5. **Recognition:** Credit added to project

### Thank You!

Every contribution, no matter how small, makes this resource better for learners worldwide. You're helping democratize AI education!

**Together, we can make AI accessible to everyone.**

---

*Questions about contributing? Open an issue with the "question" label!*

---

## Roadmap

### Current Status: Core Content Complete
The complete written tutorial is split into chapter files listed in the [course contents](#course-contents)

### Coming Soon:
- [ ] Interactive web version with navigation
- [ ] PyTorch implementation with step-by-step comments
- [ ] Visual diagrams for each chapter
- [ ] Video walkthroughs of key concepts
- [ ] Interactive attention visualizations
- [ ] Jupyter notebooks with executable examples
- [ ] Exercise sets with solutions
- [ ] Translation to other languages

Want to help with any of these? [Open an issue](../../issues) or submit a PR!

---

## Related Resources

### Essential Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original transformer paper (2017)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - The GPT-3 paper
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - InstructGPT/ChatGPT

### Advanced / Emerging
- [Instruction-Level Weight Shaping (ILWS)](https://arxiv.org/abs/2509.00251) - A newer framework for self-improving AI agents and efficient adaptation (2025)

### Complementary Tutorials
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [nanochat](https://github.com/karpathy/nanochat) - Complete end-to-end ChatGPT pipeline by Andrej Karpathy (~8K lines of code, trains in 4 hours on 8×H100)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Line-by-line implementation

### Video Courses
- [Transformers Explained by Serrano.Academy](https://www.youtube.com/watch?v=OxCpWwDCDFQ) - Excellent visual walkthrough of transformer architecture
- [Neural Networks Series by 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - Beautiful mathematical intuition for neural networks
- [MIT Introduction to Deep Learning](https://introtodeeplearning.com/) - Comprehensive deep learning course with lectures and labs
- [Stanford CS224N](http://web.stanford.edu/class/cs224n/) - Natural Language Processing with Deep Learning
- [Fast.ai](https://www.fast.ai/) - Practical Deep Learning for Coders

---

## Acknowledgments

This tutorial stands on the shoulders of giants:
- The original Google Brain team for the transformer architecture
- OpenAI for GPT and the insights into scaling laws
- Anthropic for Claude and research on AI safety
- Andrej Karpathy for making AI education accessible
- Luis Serrano (Serrano.Academy) for exceptional visual explanations of transformers
- Grant Sanderson (3Blue1Brown) for beautiful mathematical intuition in neural networks
- MIT 6.S191 team for their comprehensive Introduction to Deep Learning course
- The entire ML research community for open research

Special thanks to everyone who has provided feedback, found bugs, or contributed improvements!

---

## Contact & Community

- **Author:** Rimom Costa - [rimomcosta@gmail.com](mailto:rimomcosta@gmail.com)
- **Issues:** Found a bug or have a suggestion? [Open an issue](../../issues)
- **Discussions:** Questions or want to chat? [Start a discussion](../../discussions)

---

## License

MIT License

Copyright (c) 2025 Rimom Costa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Support This Project

If this tutorial helped you:
- **Star this repository** to help others find it
- **Share it** with your network
- **Contribute** improvements or corrections
- **Support** the author (sponsorship options coming soon)

Together, we can make AI education accessible to everyone!

---

<div align="center">

**Built with love for the AI learning community**

*"The best way to understand transformers is to build one yourself"*

[Back to Top](#transformers-for-absolute-dummies)

</div>
