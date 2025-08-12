# test_vgg.py
import tensorflow as tf

# Set memory growth (important!)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("Attempting to load VGG19...")
try:
    # Use the same weights file to replicate the error
    vgg = tf.keras.applications.VGG19(
        include_top=False,
        weights="./model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
    )
    print("VGG19 loaded successfully!")
    vgg.summary()
except Exception as e:
    print(f"Failed to load VGG19: {e}")