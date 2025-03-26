### **Lecture Notes: Introduction to Neural Networks, Week 2**  

These lecture notes cover key topics in optimization, MLPs, backpropagation, and convolutional networks.

---

## **1. Optimization Algorithms**
For a great review on how to choose parameters, hyperparameters, and diagnosing the training of your deep learning model, check out [this course from CS321](https://cs231n.github.io/neural-networks-3/#baby).
### **Gradient Descent & Stochastic Gradient Descent (SGD)**  
#### Why use gradient descent?  
- Many machine learning models optimize a loss function $L(\theta)$, where $\theta$ are the model parameters.  
- Gradient descent iteratively updates $\theta$ to minimize $L(\theta)$.  
- Update rule (full-batch gradient descent):  
 $$\theta \leftarrow \theta - \eta \nabla L(\theta)$$  
  where $\eta$ is the learning rate.  

#### Why use mini-batch SGD instead of full-batch gradient descent?  
- Computing the gradient over the entire dataset can be **slow** and **memory-intensive**.  
- SGD updates parameters using only one example at a time:  
   - choose sample $i$ at random
   - update $$\theta \leftarrow \theta - \eta \nabla L(\theta; x_i, y_i)$$  
- **Mini-batch SGD** strikes a balance: it computes gradients over small batches (e.g., 32-256 samples) to improve efficiency and stability.

#### Improvements for SGD:
The following algorithms are improvements for common pitfalls of SGD. They will not be in the quiz, but you should know that they exist.

#### Momentum
- Instead of updating directly with the gradient, accumulate a velocity term:  
 $$v_t = \beta v_{t-1} + (1 - \beta) \nabla L(\theta)$$  
 $$\theta \leftarrow \theta - \eta v_t$$  
- **Effect**: Helps escape sharp minima and reduces oscillations.

#### Improvements for SGD: Adaptive Learning Rates (Adagrad, RMSProp, Adam)  
- **Adagrad**: Adapts learning rate for each parameter based on past gradients.  
- **RMSProp**: Keeps an exponentially decaying average of past squared gradients to adjust learning rates.  
- **Adam**: Combines momentum and RMSProp for fast convergence and stability.

  [Visual comparisons](https://imgur.com/a/visualizing-optimization-algos-Hqolp) of convergence speed, escaping saddle points

---

## **2. Multilayer Perceptrons (MLPs) and Backpropagation**  
### **Stacking Linear Layers: The MLP**  
- A simple MLP consists of stacked **linear layers** (matrix multiplications) with **activation functions** (e.g., ReLU, sigmoid).
For two layers: $$MLP(x) = \sigma(W_2 \sigma(W_1 x + b_1) + b_2)$$
- Example architecture for digit classification:  
 $$\text{Input} \rightarrow \text{Linear} (784 \to 128) \rightarrow \text{ReLU} \rightarrow \text{Linear} (128 \to 10) \rightarrow \text{Softmax}$$

### **How Are Gradients Computed?**  
- Compute the gradient of the **loss function** with respect to each parameter using **backpropagation**.  

### **Backpropagation Algorithm**
1. **Forward Pass**: Compute the output and loss.  
2. **Backward Pass**: Compute gradients using the chain rule.  
  $$\frac{dL}{dW} = \frac{dL}{d\hat{y}} \cdot \frac{d\hat{y}}{dW}$$  
3. **Parameter Update**:  
  $$W \leftarrow W - \eta \frac{dL}{dW}$$

### **PyTorch Computation Graph**  
- PyTorch **automatically** builds a computation graph when performing tensor operations.  
- Calling `.backward()` computes gradients for all parameters in the graph.  
- Example:  
  ```python
  import torch
  x = torch.tensor(2.0, requires_grad=True)
  y = x**2
  y.backward()  # dy/dx = 2x
  print(x.grad)  # Output: tensor(4.)
  ```

### **PyTorch Training Workflow**  
1. **Forward pass**: Compute predictions.  
2. **Backward pass**: Compute gradients.  
3. **Optimization step**: Update parameters.  
Example:  
```python
for epoch in range(num_epochs):
    optimizer.zero_grad()   # Reset gradients
    y_pred = model(X)       # Forward pass
    loss = criterion(y_pred, y)  # Compute loss
    loss.backward()         # Backward pass (compute gradients)
    optimizer.step()        # Update weights
```

---

## **3. Convolutions: CNNs and Applications**  
### **Simple Case: 2D Convolution (1 Channel, 1 Filter)**  
- A **convolution** applies a filter (kernel) over an input image.  
- Each filter is a small matrix (e.g., \(3 \times 3\)) that slides over the input and computes a dot product.  

### **General Case: Multiple Input Channels & Filters**  
- Each **input channel** has its own filter, and multiple filters extract different features.  
- Output feature maps are summed across channels.

### **1D Convolution for Text Data**  
- A **1D convolution** can process sequential data like text.  
- A sequence (e.g., words, characters) is represented as a matrix, where each row is a word embedding or one-hot encoded letter.  
