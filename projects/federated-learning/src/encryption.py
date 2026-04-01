import numpy as np

class EncryptionSimulator:
    """
    A class to simulate encryption and decryption operations.

    This simulator adds and removes random noise to mimic the effect of
    encryption and decryption without implementing actual cryptographic algorithms.

    Attributes:
        key_length (int): The simulated key length for encryption.
    """
    def __init__(self, key_length=2048):
        """
        Initialize the EncryptionSimulator.

        Args:
            key_length (int, optional): The simulated key length. Defaults to 2048.
        """
        self.key_length = key_length

    def encrypt(self, data):
        """
        Simulate encryption by adding random noise to the data.

        Args:
            data (numpy.ndarray): The data to be "encrypted".

        Returns:
            numpy.ndarray: The "encrypted" data with added noise.
        """
        return data + np.random.normal(0, 0.01, data.shape)

    def decrypt(self, encrypted_data):
        """
        Simulate decryption by removing the added noise from the data.

        Args:
            encrypted_data (numpy.ndarray): The "encrypted" data to be decrypted.

        Returns:
            numpy.ndarray: The "decrypted" data with noise removed.
        """
        # Simulate decryption by removing the added noise
        return encrypted_data - np.random.normal(0, 0.01, encrypted_data.shape)

def encrypt_vector(encryption_simulator, vector):
    """
    Encrypt a vector using the provided encryption simulator.

    Args:
        encryption_simulator (EncryptionSimulator): The encryption simulator to use.
        vector (list or numpy.ndarray): The vector to be encrypted.

    Returns:
        list: A list of encrypted values.
    """
    return [encryption_simulator.encrypt(item) for item in vector]

def decrypt_vector(encryption_simulator, vector):
    """
    Decrypt a vector using the provided encryption simulator.

    Args:
        encryption_simulator (EncryptionSimulator): The encryption simulator to use.
        vector (list or numpy.ndarray): The vector to be decrypted.

    Returns:
        list: A list of decrypted values.
    """
    return [encryption_simulator.decrypt(item) for item in vector]