Source: https://towardsdatascience.com/what-is-residual-connection-efb07cab0d55#:~:text=Residual%20connection%20provides%20another%20path,for%20layer%20i%20by%20x.
Date: 2023-02-10
Time: 13:59
Tags: #machine_learning , #ANN,

```
Residual connection provides another path for data to reach latter parts of the neural network by skipping some layers. Consider a sequence of layers, layer _i_ to layer _i + n_, and let _F_ be the function represented by these layers. Denote the input for layer _i_ by _x_.
```

![[Pasted image 20230210140208.png]]

Residual layer is an added layer that by pass certain layers and directly goes to the destination layer. And at the destination layer, the result is an element-wise addition of both  _F(x)_ (the result of input _x_ after go through the said layers, and the original _x_). _I think the benefit of using a residual layer is to, first avoid vanishing/exploding gradient, second add identity function back to the network, as identity function turns out to be a hard function to learn by multi-layer neural networks._

**What if the dimension/size of _F(x)_ and _x_ does not match?**
==If their dimensions are different, we can replace the identity mapping with a linear transformation (i.e. multiplication by a matrix _W_), and perform _F_(_x_) + _Wx_ instead.==

```
The main reason for emphasizing the seemingly superfluous identity mapping in the figure above is that it serves as a placeholder for more complicated functions if needed. For example, the element-wise addition _F_(_x_) + _x_ makes sense only if _F_(_x_) and _x_ have the same dimensions. If their dimensions are different, we can replace the identity mapping with a linear transformation (i.e. multiplication by a matrix _W_), and perform _F_(_x_) + _Wx_ instead.
```

### Implications of using residual layers
If we add one residual layer to each one of the normal layers, then instead of one single path (the feedforward path) for the input data to go through, the input _x_ can go through multiple paths (as it can be added back at any given layer in this architecture), which practically creates an ensemble of shallow networks.

==Then the question is, does this mean that an ensemble of shallow networks perform better than a very deep network? If so, why?==

```
[1] shows that for deep feedforward neural networks, the gradients resemble white noise. This is bad for training, and the problem is called shattered gradients problem. Residual connection tackles this by introducing some spatial structure to the gradients, thus helping the training process.
```
  [1]  [The Shattered Gradients Problem: If resnets are the answer, then what is the question?](https://arxiv.org/abs/1702.08591)
