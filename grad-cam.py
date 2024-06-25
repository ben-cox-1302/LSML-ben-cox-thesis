import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tensorflow.keras.models import load_model
import h5py
import random
from skimage.transform import resize

def load_image_from_hdf5_by_class(file_path, dataset_name, class_label):
    with h5py.File(file_path, 'r') as h5f:
        X = h5f[f'X_{dataset_name}'][:]
        Y = h5f[f'Y_{dataset_name}'][:]

        Y = np.argmax(Y, axis=1)
        indices = np.where(Y == class_label)[0]

        if len(indices) == 0:
            raise ValueError("No images found for the specified class.")

        selected_index = random.choice(indices)
        image = X[selected_index].astype(np.float32)

        return image

def apply_grad_cam(model_path, image, category_index, layer_name):
    model = load_model(model_path)
    image_batch = np.expand_dims(image, axis=0)

    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
    def loss(output):
        return output[:, category_index]

    cam = gradcam(loss, seed_input=image_batch, penultimate_layer=layer_name)
    cam = normalize(cam)
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    return heatmap

def overlay_heatmap_on_image(heatmap, original_image, alpha=0.6):
    if heatmap.shape != original_image.shape[:2]:
        heatmap = resize(heatmap, original_image.shape[:2], preserve_range=True)
    overlay_image = np.uint8((1 - alpha) * original_image + alpha * heatmap)
    return overlay_image, heatmap

# Example usage:
data_file_path = '/media/bdc-pc/14A89E95A89E74C8/git_repos/data/data_xy_split/20240515_102553-2000_sample_9_class_new/split_processed_data.h5'
model_file_path = '/home/bdc-pc/git_repos/LSML-ben-cox-thesis/models/sem1_model3.h5'
layer_name = 'conv2d_layer5'
class_label = 8
dataset_name = 'test'

image = load_image_from_hdf5_by_class(data_file_path, dataset_name, class_label)
heatmap = apply_grad_cam(model_file_path, image, class_label, layer_name)
overlay_image, heatmap = overlay_heatmap_on_image(heatmap, image)

fig, ax = plt.subplots()
cax = ax.imshow(overlay_image, cmap='gray')
fig.colorbar(cm.ScalarMappable(cmap='jet'), ax=ax, orientation='vertical')
ax.axis('off')
plt.show()