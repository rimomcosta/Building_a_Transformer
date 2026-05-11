## Chapter 20: Quick Quizzes (Test Yourself!)

### Quick Check

1. **Why do we add position signals?**
   - So the model knows "I love pizza" is different from "pizza love I."

2. **What's attention like?**
   - A word looking around the sentence and listening most closely to the words that matter most.

3. **Why do residual connections help?**
   - They preserve the original signal and give gradients a direct path backward through deep stacks.

4. **What does dropout do?**
   - It randomly removes some pathways during training so the model learns more robust patterns instead of relying too heavily on one path.

5. **What's softmax?**
   - It turns raw scores into a probability distribution that adds up to 100%.

6. **Why scale attention by $\sqrt{d_k}$?**
   - To keep dot products from getting too large and making softmax overly extreme.

7. **What's the computational complexity of self-attention?**
   - Roughly $O(n^2 d)$, where $n$ is sequence length and $d$ is dimension.

8. **Why is LayerNorm useful in transformers?**
   - It normalizes each token's features independently, which keeps activations stable even when sequence lengths and batch contents vary.

9. **What's the purpose of multi-head attention?**
   - It lets the model look for multiple kinds of relationships at the same time, then combine those perspectives.

10. **Which transformer architecture did we learn?**
   - Decoder-only (GPT-style), used for text generation, chatbots, and code completion.

11. **What's the main difference between encoder and decoder?**
   - Encoders can see the full input at once; decoders only see the past and generate step by step.

12. **When would encoder-decoder be useful?**
    - When you want to read one complete input and produce a different output, such as translation, summarization, or text transformation.

13. **What's the difference between pre-training and fine-tuning?**
    - Pre-training builds general language ability; fine-tuning adapts that ability to a specific task or domain.

14. **How is a ChatGPT-style assistant different from a base GPT model?**
    - ChatGPT adds instruction tuning and alignment training so it follows requests and behaves more helpfully.

15. **Do you need millions of dollars to use transformers for your project?**
    - No. Training frontier models from scratch is expensive, but using or adapting existing models is much more accessible.

---

**Course navigation:** [Previous: Chapter 19 - Three Transformer Architectures](chapter-19-three-transformer-architectures.md) | [Next: Chapter 21 - Going Further](chapter-21-going-further.md)
