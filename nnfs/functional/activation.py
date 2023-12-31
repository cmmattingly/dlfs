import numpy as np

def tanh(z):
    """Tanh activation function

    Args:
        z: np.array

    Returns: 
        np.array
    """
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def softmax(z): 
    """Softmax activation function
    
    Args:
        z: np.array

    Returns:
        np.array
    """
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z, axis=0)

def sigmoid(z):
    """Sigmoid activation function

    Args:
        z: np.array

    Returns:
        np.array
    """
    return 1 / (1 + np.exp(-z))

def reLU(z):
    """ReLU activation function

    Args:
        z: np.array

    Returns:
        np.array
    """
    return np.max(0, z)

def leaky_reLU(z):
    """Leaky ReLU activation function

    Args:
        z: np.array

    Returns:
        np.array
    """
    return np.max(0.1 * z, z)

def selu(z, alpha, lam):
    """SELU activation function
    
    Args:
        z: np.array

    Returns:
        np.array
    """
    return np.where(z <= 0, lam * alpha * (np.exp(z) - 1), lam * z)

def elu(z, alpha):
    """ELU activation function (selu where scale = 1)

    Args:
        z: np.array

    Returns:
        np.array
    """
    return selu(z, alpha, 1)  

