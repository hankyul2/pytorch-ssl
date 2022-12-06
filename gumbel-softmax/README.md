# GUMBEL Softmax

My questions

1. Why do we add gumbel distribution?
   1. Gumbel distribution is looks like noise. How does this gumbel distribution work in here?
      - True. Gumbel distribution is added to give a noise. 
        Sampling like this is called Gumbel-max trick (see [here](http://amid.fish/humble-gumbel)).
      - Then, why should we give a noise to the distribution? This is helpful when you are interested in
        sampling categorical values according to the distribution. 
      - Why not use other noise? This is good question. You can use other distribution as noise.
        But if you use other distribution (e.g. normal), the sample distribution won't follow the original
        distribution. That's why gumbel-max trick is used so that the sample distribution will follow the original ones.
   2. Alternatives:
      - softmax with temperature: It will work of course but, there is some limitation in using the normal softmax. 
        Normal softmax will always give you the same sample, not following the original logit distribution.
2. Why do we categorize it?
   1. We couldn't answer this question by only reading this one paper. If you are more interested, please read dVAE paper.
3. Why do we need to random-sample?