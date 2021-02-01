# Brainstorm 
* Encode multiple window sizes into a single vector. Suppose given raw input matrix $\mathbf X$ and windows $d\in D$, suppose $X^{(d)}_i$ represents window $d$ of vector $\vec x_i\in\mathbf X$. (...)
* Using chroma circle, logarithmic pitch and circle of fifths seems easy enough to implement. Maybe we also want to encode note duration somehow, for me still unclear how thesis uses this. 
* Using 5D vector and window encodings might result in very high dimensional dataset with low $N$. Maybe PCA is a good idea.
* Window encodings during music generation might prove tricksy
* We need to look at how the output is going to be a probability vector; how can we train that, what kind of distribution are we looking at, etc etc
* 