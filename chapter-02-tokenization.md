## Chapter 2: Tokenization (Chopping Text into Pieces)

### Understanding the Problem

Let's say we have our vocabulary of 50,000 tokens. Simple case first:

Input text: `"I love pizza"`

**Step 1:** Look up each word in our vocabulary
- "I" → 123 ✓ Found it!
- "love" → 567 ✓ Found it!
- "pizza" → 999 ✓ Found it!

Result: `[123, 567, 999]`

Easy! But what happens with a word not in our dictionary?

**The Problem:** What about "cryptocurrency"? This word wasn't common when we built our vocabulary, so it's not in our dictionary. We can't just skip it—we need to represent every word somehow.

**Bad Solution #1:** Assign it a special "unknown" token
- Problem: All unknown words become identical to the model! "cryptocurrency" and "philosophy" would look the same.

**Bad Solution #2:** Add every possible word to the vocabulary
- Problem: English has millions of words, plus new words invented daily ("selfie," "podcast," "ChatGPT"). Our vocabulary would be infinite!

**Good Solution:** Break unknown words into known pieces!

### The Byte-Pair Encoding (BPE) Solution

BPE is like a smart puzzle solver. When it sees "cryptocurrency":

1. Check: Is "cryptocurrency" in the vocabulary? No.
2. Try: Can we break it into two pieces we know? "crypto" + "currency"? Yes! Both are common enough to be in our vocabulary.
3. Result: "cryptocurrency" → [token for "crypto", token for "currency"]

If a piece still isn't in the vocabulary, break it down further:
- "cryptocurrency" → ["crypto", "currency"] → ["crypt", "o", "cur", "ren", "cy"]

**Important simplification note:** For this tutorial, we're using a **character-level mental model** because it's much easier to understand on paper. In many real GPT-style systems, tokenizers are actually **byte-level**, not letter-level. The big idea is the same: if a full word isn't available, the tokenizer keeps breaking it into smaller known pieces until it finds pieces that ARE in the vocabulary.

So in our toy explanation, the worst case is "break down to characters." In many production tokenizers, the worst case is closer to "break down to bytes."

Think of it like describing something using words someone knows. If a child doesn't know "automobile," you might say "auto" and "mobile" (self-moving). If they don't know "auto," you break it down further to basic concepts they do understand.

### How BPE Builds the Vocabulary (The Training Process)

Before we can use BPE, we need to create our 50,000-token vocabulary. Here's how:

**Step 1: Start with basic pieces**
In our simplified tutorial version, begin with just the individual characters: a, b, c, ..., z, A, B, C, ..., Z, plus space and punctuation.
That's about 100 basic tokens in our toy setup.

**Step 2: Find the most common pair**
Look through millions of example texts. Which two characters appear next to each other most often?
Maybe "t" followed by "h" appears 150,000 times in your text.

**Step 3: Merge them**
Create a new token "th" and add it to your vocabulary. Now you have 101 tokens.
Replace all instances of "t" + "h" with the single token "th".

**Step 4: Repeat**
Find the next most common pair. Maybe now it's "e" + "r" appearing 120,000 times.
Create token "er". Now you have 102 tokens.

**Step 5: Keep going**
Continue this process. After many merges, you'll have common subwords:
- "ing" (from "i" + "n" + "g" merging over multiple steps)
- "tion" (common ending)
- "the" (very common word)
- Eventually full common words: "pizza", "computer", "love"

**Step 6: Stop at 50,000**
In our toy setup, if you started from about 100 basic pieces, you'd need **roughly** 49,900 merges to end up near 50,000 tokens.

**Production reality:** Real tokenizers often have extra reserved tokens, different starting alphabets/byte sets, and implementation details that make the exact count messier. The important idea is NOT the exact arithmetic. The important idea is: keep merging common pairs until you reach your target vocabulary size.

**The beautiful result:** Common words are single tokens (efficient!), while rare words are broken into meaningful pieces (flexible!).

Example hierarchy in the final vocabulary:
```
Character level: "a", "b", "c"
Subword level:   "th", "ing", "er"  
Word level:      "the", "pizza", "love"
```

**Why this works:** Language has patterns. The character pair "th" appears together constantly in English, so it deserves its own token in our simplified example. But "qz" almost never appears together, so we don't waste a token on that combination. In a production tokenizer, the same principle applies even if the smallest units are bytes instead of letters.

**Deterministic and reversible:** Once we've built our vocabulary, tokenization is completely deterministic—the same text always produces the same tokens. And we can always convert tokens back to text perfectly (unlike lossy compression). It's a two-way mapping.

---

**Course navigation:** [Previous: Chapter 1 - Building Our Vocabulary](chapter-01-building-our-vocabulary.md) | [Next: Chapter 3 - Embeddings](chapter-03-embeddings.md)
