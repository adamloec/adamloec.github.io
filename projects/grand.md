---
title: Grand - Neural Network Framework
layout: home
nav_order: 1
parent: Projects
---

# Grand Framework
{: .no_toc }
{: .fs-9 }
A naive Neural Network and Tensor library, for learning purposes.
{: .fs-6 .fw-300 }

[Github][grand github]{: .btn .fs-5 .mb-4 .mb-md-0 }

## Table of contents
{: .no_toc .text-delta }
1. TOC
{:toc}

---

## About

The Grand Framework is meant to be a personal learning project for neural network model architecture, mathematics, and training. My intention is to showcase what I have learned, as well as deepen my understanding, in regards to machine learning and deep neural networks. The project started as a C++/CUDA based implementation, but after falling in love with Python, it was soon converted.

The framework is built on Numpy, a popular Python library for matrices and multi-dimensional arrays that is commonly used in the field of machine learning. CUDA kernels will eventually be used to transfer training operations to the GPU for faster runtime.

### Goals

1. Design a working Tensor object with relevent utilities and functions.
2. Implement CUDA kernels for running operations on a NVIDIA GPU.
3. Design layer and model objects to construct a working MLP (Multilayer Perceptron) neural network, built on custom Tensor objects.
4. Implement forward passing, utilizing linear regression and common activation functions.
    - Designed to allow for use with Tensorflow Datasets.
5. Implement back propogation, utilizing common cost/loss functions and optimizers.
6. Final test: Build a functional model, running on either CPU or GPU, trained on the MNIST hand-drawn digits dataset with **loss < 10%**.

### Readings, resources, and design inspirations

- Machine Learning Specialization coursework - Andrew Ng
- Deep Learning with Python, Second Edition - Francois Chollet 
- Neural Networks from Scratch (NNFS) - Harrison Kinsley (Sentdex)

---

## Research

---

## Development

### Dependencies

- Python
    - Numpy
    - Python CUDA
- CUDA / NVCC

### Tensors

Tensors are the backbone of any neural network framework. These objects contain the data, dimensionality, while allowing for complex mathematical operations.

##### Design
{: .mb-lg-4 }

The design choices for the Tensor object were simple. I wanted the user to be able to pass in any type of array, list, or np.ndarray to the constructor to initialize new Tensors. This allows the object to be intuitive, and easy to implement.

{% highlight Python %}
class Tensor:
    """
    Name: Tensor()
    Purpose: Parent constructor for the Tensor object.
    Parameters: data (list, np.ndarray), dtype (Any)
    Return: None

    Uses: \n
    t = Tensor([1, 2, 3])
    t = Tensor(np.array([1, 2, 3]))
    """
    def __init__(self, data, dtype=np.float32, device=0):

        if not isinstance(data, (np.ndarray, list)):
            raise TypeError("ERROR: Data must be of type list, np.ndarray")
        data = np.array(data, dtype=dtype)
        self.data = data
        self.shape = self.data.shape
        self.dtype = dtype
{% endhighlight %}

The tensor object also contains built-in class methods for different types of Tensors (Similar to Numpy). These methods take dimensions as parameters and construct a new Tensor object with specific values dependent on the method:
- Zeros, a Tensor initialized with Zeros.
- Ones, a Tensor initialized with Ones.
- Random, a Tensor initialized with random values.
- Empty, a Tensor initialized with NaN values.

Below is an example of this implementation:

{% highlight Python %}
@classmethod
def zeros(cls, *shape, dtype=np.float32):
    """
    Name: Tensor.zeros()
    Purpose: Class method for creating Tensor object with zeros, given shape.
    Parameters: *shape (N number of integers), dtype (Any)
    Return: Tensor

    Uses: \n
    t = Tensor.zeros(1, 2)
    """

    for d in shape:
        if not isinstance(d, int):
            raise TypeError("ERROR: shape must be of type int")
    t = cls(data=np.zeros(shape, dtype=dtype), dtype=dtype)
    return t
{% endhighlight %}

These are helpful for initializing weights, biases, and inputs in layers prior to starting training.

##### Operations
{: .mb-lg-4 }

Tensors need to be able to perform operations with one another, like matrix multiplcation and addition, so that we can perform linear regression on nodes in the neural network. This functionality can be built into Python classes and executed intuitively.

For example, matrix addition:

{% highlight Python %}
def __add__(self, b):
    """
    Name: __add__()
    Purpose: Addition function, between 2 Tensors or Tensor and integer/float
    Parameters: b (Tensor, integer, float)
    Return: Tensor

    Uses: \n
    Tensor = Tensor + Tensor
    Tensor = Tensor + integer/float
    """

    if isinstance(b, Tensor):
        assert self.shape == b.shape, "ERROR: A and B input Tensors must be the same size"
        return Tensor(self.data + b.data)
    elif isinstance(b, (int, float)):
        return Tensor(self.data + b)
    else:
        raise TypeError("ERROR: Data B must be of type Tensor, int, float")
{% endhighlight %}

##### Testing
{: .mb-lg-4 }

I created a custom test bench to verify that the Tensor operations were error free, as well as comparing the results to Numpy equivalent operations.

The testbench constructor takes in a function, or the operation that will be tested, as a parameter. Below is the Testbench class, a function operation that will be executed in the testbench, and the resulting output:

{% highlight Python %}
class Testbench:
    """
    Name: Testbench()
    Purpose: Constructor for the Testbench object.
    Parameters: func (mathematical operation function), runs (int)
    Return: None

    Uses: \n
    Testbench(tensor_tensor_add)
    """
    def __init__(self, func, runs=1000):
        self.func = func
        self.runs = runs

        tic = time.perf_counter()
        for run in range(runs):

            dim1, dim2 = random.randint(1, 100), random.randint(1, 100)
            np1 = np.array(np.random.rand(dim1, dim2), dtype=np.float32)
            np2 = np.array(np.random.rand(dim1, dim2), dtype=np.float32)

            if not func(dim1=dim1, dim2=dim2, np1=np1, np2=np2):
                print(f"TEST {func.__name__} {run}/{runs} {Colors.FAIL}FAIL{Colors.ENDC}")

            if run % 100 == 0:
                print(f"TEST {func.__name__} Runs: {run}/{runs} {Colors.OKGREEN}PASS{Colors.ENDC}")
        toc = time.perf_counter()
        
        print(f"TEST {func.__name__} Runs: {runs}/{runs} {Colors.OKGREEN}PASS{Colors.ENDC}") 
        print(f"Duration: {toc - tic:0.4f}s")
{% endhighlight %}

{% highlight Python %}
def tensor_tensor_add(**kwargs):
    """
    Name: tensor_tensor_add()
    Purpose: Test function for addition between 2 tensor's.
    Parameters: **kwargs(dim1, dim2, np1, np2)
    Return: Bool

    Uses: \n
    Intended to be used inside of Testbench(func)
    """
    tensor_result = np.around((Tensor(kwargs['np1']) + Tensor(kwargs['np2'])).data, decimals=4)
    np_result = np.around(kwargs['np1'] + kwargs['np2'], decimals=4)
    return (tensor_result == np_result).all()

Testbench(tensor_tensor_add)
{% endhighlight %}

```
TEST tensor_tensor_add Runs: 0/1000 PASS
TEST tensor_tensor_add Runs: 100/1000 PASS
TEST tensor_tensor_add Runs: 200/1000 PASS
TEST tensor_tensor_add Runs: 300/1000 PASS
TEST tensor_tensor_add Runs: 400/1000 PASS
TEST tensor_tensor_add Runs: 500/1000 PASS
TEST tensor_tensor_add Runs: 600/1000 PASS
TEST tensor_tensor_add Runs: 700/1000 PASS
TEST tensor_tensor_add Runs: 800/1000 PASS
TEST tensor_tensor_add Runs: 900/1000 PASS
TEST tensor_tensor_add Runs: 1000/1000 PASS
Duration: 0.2440s
```

---


[grand github]: https://github.com/adamloec/grand