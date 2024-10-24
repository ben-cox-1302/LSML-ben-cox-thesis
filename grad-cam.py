import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs except for errors
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras

# Display
import matplotlib as mpl
import matplotlib.pyplot as plt


def load_csv_as_np_array(file_path):
    try:
        data = np.loadtxt(file_path, delimiter=',', dtype=np.float64)
        # data = remove_cosmic_rays(data, 1)
        data = np.expand_dims(data, axis=-1)
        return data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def get_img_array(img_path):
    # `img` is a PIL image of size 299x299
    img = load_csv_as_np_array(img_path)
    plt.imshow(img)
    plt.show()
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Check if model.output is a list and take the appropriate output
    if isinstance(model.output, list):
        output = model.output[0]
    else:
        output = model.output

    # Create a model that maps the input image to the activations of the last conv layer and the output predictions
    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, chem_name, alpha=0.4):
    # Load the original image
    img = load_csv_as_np_array(img_path)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    save_path = 'plots/' + chem_name

    # Create a Matplotlib figure
    plt.figure(figsize=(8, 8))

    # Display the superimposed image
    plt.imshow(superimposed_img)
    plt.axis('off')  # Hide the axes for a cleaner image

    title = f'Grad Cam: {chem_name}'

    plt.title(title, fontsize=16)  # You can adjust the fontsize if needed

    # Save the plot with the superimposed image and title
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    # Close the figure to prevent display
    plt.close()


def remove_cosmic_rays(data, threshold_factor=5):
    """
    Remove cosmic rays by replacing bright pixels that are significantly
    brighter than their neighbors.

    :param data: A 2D numpy array containing the image data.
    :param threshold_factor: The factor used to determine if a pixel is
                             significantly brighter than the median of its neighbors.
    :return: A 2D numpy array with cosmic rays removed.
    """
    # Create a copy of the data to modify
    cleaned_data = np.copy(data)

    # Calculate the median of the entire data for a basic thresholding
    median_value = np.median(data)

    # Calculate the median filtered version of the original data
    # This uses a 3x3 neighborhood by default
    from scipy.ndimage import median_filter
    median_filtered = median_filter(data, size=3)

    # Find potential cosmic rays
    # These are pixels where the data is significantly greater than the median filter result
    cosmic_rays = data > median_filtered + threshold_factor * np.abs(median_filtered - median_value)

    # Replace these pixels with the median of their 3x3 neighborhoods
    cleaned_data[cosmic_rays] = median_filtered[cosmic_rays]

    return cleaned_data


base_path = '/media/benjamin/14A89E95A89E74C8/git_repos/data/data_raw/grad_cam/'

chem_paths = {
    'Lithium-Niobate' : base_path + '07-06-24-LithiumNiobate_1pulse_1000acq/07-06-24-LithiumNiobate_1pulse_1000acq5.csv',
    'Paracetamol' :     base_path + '28-03-24-Paracetamol_1pulse_2000acq_BC/28-03-24-Paracetamol_1pulse_2000acq_BC5.csv',
    'Bi-Carb' :         base_path + '05-04-24-Bicarb_1pulse_2000acq_BC/05-04-24-Bicarb_1pulse_2000acq_BC5.csv',
    'Deep-Heat' :       base_path + '05-04-24-DeepHeat_1pulse_2000acq_BC/05-04-24-DeepHeat_1pulse_2000acq_BC5.csv',
    'Water' :           base_path + '04-04-24-Water_1pulse_2000acq_BC/04-04-24-Water_1pulse_2000acq_BC5.csv',
    'Barium' :          base_path + '05-04-24-Barium_1pulse_2000acq_BC/05-04-24-Barium_1pulse_2000acq_BC5.csv',
    'Erythritol' :      base_path + '28-03-24-Erythritol_1pulse_2000acq_BC/28-03-24-Erythritol_1pulse_2000acq_BC5.csv',
    'Citric-Acid' :     base_path + '28-03-24-CitricAcid_1pulse_2000acq_BC/28-03-24-CitricAcid_1pulse_2000acq_BC5.csv',
    'Flour' :           base_path + '04-04-24-Flour_1pulse_2000acq_BC/04-04-24-Flour_1pulse_2000acq_BC5.csv'
}

model_path = '/home/benjamin/git_repos/LSML-ben-cox-thesis/models/cleanData_2D_gradCam.keras'
last_layer_name = 'conv2d_layer5'

model = tf.keras.models.load_model(model_path)

# Remove last layer's softmax
model.layers[-1].activation = None

for name, path in chem_paths.items():
    print(f'Processing {name}')
    chem_img = get_img_array(path)
    # Print what the top predicted class is
    preds = model.predict(chem_img)
    predicted_class_indices = np.argmax(preds, axis=1)
    print("Predicted Class:", predicted_class_indices)
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(chem_img, model, 'conv2d_layer5')

    save_and_display_gradcam(path, heatmap, name)
