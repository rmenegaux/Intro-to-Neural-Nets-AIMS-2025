## Lecture Notes for the first two classes

### **Supervised Learning Framework**  

Supervised learning is a machine learning approach where the goal is to learn a function that maps input features **x** to target outputs **y**, based on labeled training data. Given a dataset of **n** examples:  

$$
\{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}
$$

where each $x_i \in \mathbb{R}^d$ is an input feature vector of dimension $d$, and $y_i \in \mathbb{R}$ is the corresponding label, we aim to learn a model $f_\theta(x)$ parameterized by $\theta$ that minimizes a loss function measuring the discrepancy between predictions and true labels.  

The learning process typically involves:  
1. **Selecting a hypothesis class**: Choosing a family of functions $f_\theta$ (e.g., linear models, neural networks).  
2. **Defining a loss function**: Common choices include mean squared error (MSE) for regression and cross-entropy for classification.  
3. **Optimization**: Adjusting parameters $\theta$ to minimize the loss function, often using gradient-based methods like stochastic gradient descent (SGD).  

---

### **Gradient Definition and the Chain Rule**  

The **gradient** of a function measures how it changes with respect to its input variables. if $f: \mathbb{R}^d \to \mathbb{R}$ is a differentiable function, its gradient is the vector of partial derivatives:

$$
\nabla f(x) = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right)^T
$$

It points in the direction of the steepest increase of $f$ and is fundamental for optimization. 


As an alternative (and more general) definition, The **gradient** of a function $f: \mathbb{R}^d \to \mathbb{R}$ at a point $x$ is the unique vector $\nabla f(x)$ that satisfies:

$$
f(x + h) = f(x) + \nabla f(x) \cdot h + o(\|h\|)
$$

for small perturbations $h$. $\nabla f(x)$ gives the best linear approximation of the function at $x$.  


This generalizes the concept of the derivative to multiple dimensions.  

#### **Chain Rule, short version**  

The **chain rule** is essential when computing gradients in deep learning. If we have a function composed of nested functions, such as $z = f(g(x))$, the derivative follows:

$$\frac{dz}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

(Just be careful with the dimensions of these objects: see full version at the end of the notes)

This principle allows us to compute gradients efficiently in neural networks using **backpropagation**, where gradients are propagated backward through layers using the chain rule.

---

### **Ordinary Least Squares and Gradient Computation**  

In linear regression, we estimate the outputs $y$ as a linear function of the inputs $X$ is modeled as:

$$
\hat{y} = Xw + b
$$

where:  
- $X \in \mathbb{R}^{n \times d}$ is the matrix of input features,  
- $w \in \mathbb{R}^{d}$ is the vector of weights,  
- $b \in \mathbb{R}$ is the bias term.  

To simplify notation, we omit $b$ in the following derivations by assuming an additional column of ones in $X$, making $X \in \mathbb{R}^{n \times (d+1)}$ and $w \in \mathbb{R}^{(d+1)}$.  

The objective is to minimize the **mean squared error (MSE) loss**:

$$
L(w) = \frac{1}{n} \sum_{i=1}^{n} (y_i - x_i^T w)^2
$$

This can be rewritten in matrix form as:

$$
L(w) = \frac{1}{n} \lvert y - Xw \rvert ^2
$$

To compute the gradient, we differentiate $L(w)$ with respect to $w$:

$$
\nabla_w L = \frac{2}{n} X^T (Xw - y)
$$

Setting this gradient to zero, we obtain the **optimal** weights:

$$
w^* = (X^T X)^{-1} X^T y
$$

which provides the closed-form solution for linear regression, assuming $X^T X$ is invertible.  

However, in cases where $X^T X$ is ill-conditioned or high-dimensional, regularization techniques like **Ridge Regression** (which adds an $\ell_2$ penalty to $w$) are commonly used to improve stability.  

---

### **Gradient Descent**  

In practice, most training objectives do not admit closed-form solutions. Instead, we often use **gradient descent**, an iterative optimization algorithm that updates parameters in the direction of the negative gradient. The algorithm is:

1. Randomly initialize the parameters $\theta^{(0)}$
2. **Repeat** until convergence (i.e., when $L$ stops changing significantly):

$$\theta^{(t+1)} = \theta^{(t)} - \eta \nabla_\theta L$$

where $\eta$ is the learning rate, which controls the step size. Gradient descent comes in different variants:  

- **Full Gradient Descent**: Uses the full dataset to compute the gradient at each step.  
- **Stochastic Gradient Descent (SGD)**: Uses a single random example per update, which makes training noisier but computationally cheaper.  
- **Mini-batch Gradient Descent**: Uses a small batch of examples per update, balancing efficiency and stability.  

Gradient descent is widely used in deep learning and other large-scale machine learning problems where exact solutions are infeasible.

#### Convergence
Gradient Descent converges when the algorithm reaches a point where the value of $L$ no longer changes significantly, or where its gradient is close to 0.

In convex problems (e.g., linear regression, logistic regression), gradient descent will converge to the **global minimum** of the cost function, which represents the optimal values for the parameters.
  
In non-convex problems (e.g., deep learning), gradient descent may converge to a **local minimum** or a **saddle point**. This means that while the algorithm has found a point where the gradient is zero, it may not be the absolute best solution. Advanced variants of gradient descent (like stochastic gradient descent, momentum, or Adam) can help escape local minima or saddle points.

### Factors affecting convergence:
- **Learning rate**: If the learning rate is too high, gradient descent may "overshoot" and not converge at all, jumping past the minimum. If it's too low, convergence may be extremely slow or stuck in a suboptimal local minimum.

-----
### Bonus
#### **Chain Rule, full version**
The **Jacobian** of a vector-valued function $f: \mathbb{R}^m \to \mathbb{R}^n$ is the matrix of all its partial derivatives, denoted as $J_f(x) \in \mathbb{R}^{n \times m}$, where each row corresponds to the gradient of one output component with respect to the inputs. It generalizes the gradient to functions with multiple outputs.

$$[J_f]_{ij} = \frac{\partial f_i}{\partial x_j}$$

If we have functions $f: \mathbb{R}^m \to \mathbb{R}^n$ and $g: \mathbb{R}^n \to \mathbb{R}^p$, the composition $h = g \circ f$ is a function $h: \mathbb{R}^m \to \mathbb{R}^p$. Its gradient is given by:  
$$\nabla h(x) = J_g(f(x)) J_f(x),$$  
where:  
- $J_f(x) \in \mathbb{R}^{n \times m}$ is the Jacobian of $f$,  
- $J_g(f(x)) \in \mathbb{R}^{p \times n}$ is the Jacobian of $g$,  
