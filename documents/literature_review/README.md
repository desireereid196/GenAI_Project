# Background & Literature Review

Image captioning is a challenging interdisciplinary task that lies at the intersection of computer vision and natural language processing. It involves generating coherent and contextually relevant textual descriptions for images, a capability with wide-ranging applications in accessibility, content moderation, and human-computer interaction. Early methods relied heavily on template-based or retrieval-driven techniques, which lacked flexibility and adaptability to diverse image content.

The field experienced a transformative shift with the advent of deep learning. A seminal model, Show and Tell by Vinyals et al. (2015), introduced an end-to-end trainable encoder-decoder framework. This architecture leveraged a pre-trained Convolutional Neural Network (CNN)—typically InceptionV3 or similar, to extract high-level visual features from images, which were then passed to a Long Short-Term Memory (LSTM) network that sequentially generated natural language captions. This laid the groundwork for modern captioning systems.
Building upon this foundation, Xu et al. (2015) proposed Show, Attend and Tell, incorporating an attention mechanism that allowed the decoder to dynamically focus on salient regions of the image during caption generation, improving both performance and interpretability.

Another key advancement in caption generation involves decoding strategies. While greedy decoding selects the most likely word at each time step, beam search maintains multiple candidate sequences, expanding and ranking them in parallel. This method often results in more fluent and contextually accurate captions by reducing locally optimal but globally subpar word choices.

Furthermore, pre-trained word embeddings, particularly GloVe (Pennington et al., 2014), have become standard in natural language processing tasks. These embeddings provide semantically rich vector representations that capture word relationships and can improve convergence and semantic accuracy in captioning models. In our case, we attempted to integrate GloVe into the decoder’s embedding layer, but encountered shape mismatches between the embedding vectors and the model’s vocabulary size, which ultimately prevented successful incorporation during training.

In this project, we implemented a classical CNN-RNN captioning pipeline. Image features were extracted using a frozen ResNet50 encoder, and a custom LSTM decoder was trained to generate captions. To enhance caption fluency, we implemented beam search decoding and compared its output to greedy search. We also conducted visualization and embedding analysis to assess the model’s internal semantic structure. Together, these experiments helped us benchmark classical methods while exploring enhancements in language modeling and decoding quality.

## References

* Vinyals et al. (2015). [Show and Tell: A Neural Image Caption Generator](#show-and-tell-a-neural-image-caption-generator). (Source: <https://arxiv.org/abs/1411.4555>)
* Xu et al. (2015). [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](#show-attend-and-tell-neural-image-caption-generation-with-visual-attention). (Source: <https://arxiv.org/abs/1502.03044>)
* Luo et al. (2022) [A Frustratingly Simple Approach for End-to-End Image Captioning](#a-fustratingly-simple-approach-for-end-to-end-image-captioning). (Source: <https://arxiv.org/abs/2201.12723>)
* Pennington et al. (2014). [GloVe: Global Vectors for Word Representation](#glove-global-vectors-for-word-representation). (Source: <https://aclanthology.org/D14-1162/>)

### [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)

Automatically describing the content of an image is a fundamental problem in artificial intelligence that connects computer vision and natural language processing. In this paper, we present a generative model based on a deep recurrent architecture that combines recent advances in computer vision and machine translation and that can be used to generate natural sentences describing an image. The model is trained to maximize the likelihood of the target description sentence given the training image. Experiments on several datasets show the accuracy of the model and the fluency of the language it learns solely from image descriptions. Our model is often quite accurate, which we verify both qualitatively and quantitatively. For instance, while the current state-of-the-art BLEU-1 score (the higher the better) on the Pascal dataset is 25, our approach yields 59, to be compared to human performance around 69. We also show BLEU-1 score improvements on Flickr30k, from 56 to 66, and on SBU, from 19 to 28. Lastly, on the newly released COCO dataset, we achieve a BLEU-4 of 27.7, which is the current state-of-the-art. ([Back to References](#references))

### [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)

Inspired by recent work in machine translation and object detection, we introduce an attention based model that automatically learns to describe the content of images. We describe how we can train this model in a deterministic manner using standard backpropagation techniques and stochastically by maximizing a variational lower bound. We also show through visualization how the model is able to automatically learn to fix its gaze on salient objects while generating the corresponding words in the output sequence. We validate the use of attention with state-of-the-art performance on three benchmark datasets: Flickr8k, Flickr30k and MS COCO. ([Back to References](#references))

### [A Fustratingly Simple Approach for End-to-End Image Captioning](https://arxiv.org/abs/2201.12723)

Image Captioning is a fundamental task to join vision and language,
concerning about cross-modal understanding and text generation.
Recent years witness the emerging attention on image captioning.
Most of existing works follow a traditional two-stage training
paradigm. Before training the captioning models, an extra object
detector is utilized to recognize the objects in the image at first.
However, they require sizeable datasets with fine-grained object
annotation for training the object detector, which is a daunting task.
In addition, the errors of the object detectors are easy to propagate
to the following captioning models, degenerating models’ performance.
To alleviate such defects, we propose a frustratingly simple
but highly effective end-to-end image captioning framework, Visual
Conditioned GPT (VC-GPT), by connecting the pre-trained
visual encoder (CLIP-ViT) and language decoder (GPT2). Different
from the vanilla connection method that directly inserts the crossattention
modules into GPT2, we come up with a self-ensemble
cross-modal fusion mechanism that comprehensively considers
both the single- and cross-modal knowledge. As a result, we do
not need extra object detectors for model training. Experimental
results conducted on three popular image captioning benchmarks
(MSCOCO, Flickr30k and NoCaps) demonstrate that our VC-GPT
achieves either the best or the second-best performance across all
evaluation metrics over extensive baseline systems. ([Back to References](#references))

### [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/)

Recent methods for learning vector space representations of words have succeeded in capturing fine-grained semantic and syntactic regularities using vector arithmetic, but the origin of these regularities has remained opaque. We analyze and make explicit the model properties needed for such regularities to emerge in word vectors. The result is a new global logbilinear regression model that combines the advantages of the two major model families in the literature: global matrix factorization and local context window methods. Our model efficiently leverages statistical information by training only on the nonzero elements in a word-word cooccurrence matrix, rather than on the entire sparse matrix or on individual context windowsinalargecorpus. Themodelproduces a vector space with meaningful substructure, as evidenced by its performance of 75% on a recent word analogy task. It also outperforms related models on similarity tasks and named entity recognition ([Back to References](#references))
