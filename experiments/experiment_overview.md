# Experiment Overview

1. __Baseline Training and Evaluation (Supports Research Question 1)__
Train CNN-LSTM model on Flickr8k and evaluate using automatic metrics (BLEU, METEOR, etc.)

2. __Error Analysis (Supports Research Questions 2 and 4)__
Qualitative analysis of bad predictions, noting confusing categories (i.e., missing objects, hallucinations, grammatical errors).

3. __Semantic Fidelity Comparison (Supports Research Question 3)__
Compute the correlation between human judgement and traditional metrics (e.g. BLEU and METEOR,etc.) compare it to the correlation between human judgement and embedding-based metrics (e.g., BERTScore, CLIPScore, etc.). Optional: If time permits, compute correlation between human judgement and the visual attention map alignment, then compare against traditional and embedding-based metrics.

4. __Generalization (Supports Research Question 1)__
Assess the model's ability to generalize by evaluating its captioning performance on an out-of-domain dataset (e.g., Flickr30k) without additional fine-tuning. Performance will be measured using standard metrics (BLEU, METEOR, etc.) to determine how well the model transfers to visually and semantically diverse content beyond the training distribution.

# Research Questions
1. How accurately do image captioning models preserve the semantic content of an input image?

2. Where do modern generative captioning models hallucinate or misinterpret visual content?

3. Can embedding-based metrics or visual attention maps help quantify caption fidelity beyond BLEU and METEOR scores?

4. What failure modes occur when captioning under visual ambiguity (e.g., occlusions, cluttered scenes, etc.)?