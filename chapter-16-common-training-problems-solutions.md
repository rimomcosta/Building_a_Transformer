## Chapter 16: Common Training Problems & Solutions

### The Troubleshooting Guide

**Training a transformer is like learning to ride a bicycle** - sometimes things go wrong, and you need to diagnose the problem!

**Common scenarios:**
- **The bike won't move:** Loss not decreasing (learning rate too low?)
- **The bike wobbles wildly:** Loss exploding (learning rate too high?)
- **Great on the driveway, crashes on the street:** Overfitting (memorized training, can't generalize)
- **Can't even balance:** Underfitting (model too small or simple)

**Don't panic!** Every problem has telltale symptoms and known solutions. Let's learn how to diagnose and fix them!

**Think of this chapter as your "Transformer Emergency Handbook"** - when something goes wrong during training, come back here!

### Problem 1: Loss Not Decreasing (The Stubborn Bike)

**Symptoms:** Training seems stuck, loss barely budges!
```
Epoch 1: Loss = 6.234
Epoch 2: Loss = 6.189  ← Only 0.045 improvement
Epoch 3: Loss = 6.201  ← Went up slightly!
Epoch 4: Loss = 6.178  ← Tiny improvement again
```

**It's like pedaling a bike but barely moving forward!**

**Possible causes & fixes:**
1. **Learning rate too high:** Reduce by 10× (0.0001 → 0.00001)
2. **Learning rate too low:** Increase by 2× (0.0001 → 0.0002)
3. **Bad initialization:** Reinitialize with proper Xavier scaling
4. **Gradient clipping too aggressive:** Increase threshold (1.0 → 5.0)

### Problem 2: Loss Exploding (NaN)

**Symptoms:**
```
Epoch 1: Loss = 4.567
Epoch 2: Loss = 8.234
Epoch 3: Loss = 45.678
Epoch 4: Loss = NaN
```

**Fixes:**
1. **Gradient clipping:** Clip to norm 1.0
2. **Lower learning rate:** Try 0.00001
3. **Check data:** Remove corrupted examples
4. **Layer normalization:** Ensure epsilon is small enough (1e-5)

### Problem 3: Overfitting

**Symptoms:**
```
Training loss: 1.234 ← Great!
Validation loss: 3.987 ← Terrible!
```

**Fixes:**
1. **Increase dropout:** 0.1 → 0.3
2. **Add weight decay:** 0.01
3. **More training data:** Augment or collect more
4. **Smaller model:** Reduce layers or $d_{\text{model}}$
5. **Early stopping:** Stop when validation loss plateaus

### Problem 4: Underfitting

**Symptoms:**
```
Training loss: 4.567 ← Bad
Validation loss: 4.623 ← Also bad
```

**Fixes:**
1. **Larger model:** More layers or wider dimensions
2. **Train longer:** More epochs
3. **Reduce regularization:** Lower dropout (0.3 → 0.1)
4. **Better features:** Improve tokenization or preprocessing

---

**Course navigation:** [Previous: Chapter 15 - Additional Techniques](chapter-15-additional-techniques.md) | [Next: Chapter 17 - Putting It All Together](chapter-17-putting-it-all-together.md)
