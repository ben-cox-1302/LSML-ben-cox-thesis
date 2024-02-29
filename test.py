import tensorflow as tf
import time

print("hell2o")

print('Available devices:')
devices = tf.config.list_physical_devices('GPU')
print(devices if devices else 'No GPU devices found.')

def gpu_stress_test():
    size = 8000  # Increased size for more computation intensity
    iterations = 300  # Number of times the operation will be performed
    print(f'Performing matrix multiplication with matrices of size {size}x{size}, {iterations} times...')

    for i in range(iterations):
        print(f'Iteration {i+1}...')
        A = tf.random.normal([size, size], dtype=tf.float32)
        B = tf.random.normal([size, size], dtype=tf.float32)
        start_time = time.time()
        C = tf.matmul(A, B)
        # Optionally, force TensorFlow to complete the operation with .numpy()
        C.numpy()
        end_time = time.time()
        print(f'Time taken for iteration {i+1}: {end_time - start_time} seconds')

gpu_stress_test()
