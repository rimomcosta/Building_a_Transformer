## Chapter 1: Building Our Vocabulary (The Dictionary)

Computers only speak numbers, so we create a giant dictionary mapping words to IDs. Think of it like assigning a unique ID number to every student in a huge school—instead of saying "Sarah," you could say "Student #567."

### Our Vocabulary Setup

Let's build a vocabulary of **50,000 tokens**. Why exactly 50,000?

First, understand that a "token" can be a whole word, part of a word, or even a single letter. The number 50,000 is a sweet spot:

- **Too few tokens (e.g., 10,000):** Many words wouldn't fit, so we'd break everything into tiny pieces. "Unbelievable" might become "un", "believ", "able"—harder for the model to understand
- **Too many tokens (e.g., 500,000):** We'd have a word for everything, but it creates a HUGE lookup table that's expensive to store and search through
- **50,000 is just right:** Covers most common English words plus frequent subwords, without being wasteful

Real systems use similar sizes: GPT-3 uses 50,257, and some frontier systems use vocabularies on the order of 100,000 tokens.

Here's what our vocabulary looks like:

```
Token ID    Token
--------    -----
0           <pad>     (special: padding)
1           <start>   (special: start of text)
2           <end>     (special: end of text)
...         ...
123         "I"
567         "love"
999         "pizza"
1234        "the"
2001        "cat"
5678        "running"
...         ... (up to 50,000)
```

Imagine organizing a massive library. You could shelve books by title alphabetically, but that's slow to search. Instead, you give each book a unique number—book #42 is always in the same spot. When someone wants "Pride and Prejudice," you look up its number (say, #15,234) and go straight to it. Same idea here: "pizza" is always token #999.

**The trade-off:** Smaller vocabularies mean less memory (fewer entries to store), but words get chopped into more pieces. Larger vocabularies keep words intact, but the lookup table gets huge. Modern transformers need to multiply every single vocabulary entry by their embedding size, so 50,000 × 12,288 dimensions = 614 million numbers just for the vocabulary! That's why size matters.

---

