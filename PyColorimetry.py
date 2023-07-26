import cv2
import os
import groundingdino.datasets.transforms as T
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from segment_anything import SamAutomaticMaskGenerator

from skimage.color import rgb2xyz
from skimage.color import xyz2lab
from skimage.color.colorconv import _prepare_colorarray
from numpy import sqrt, arctan2, degrees


import warnings

# Ignorar las advertencias de categoría FutureWarning y UserWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
}

CACHE_PATH = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints"))

def load_model_hf(repo_id, filename, ckpt_config_filename, device='gpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    #print(f"Model loaded from {cache_file} \n => {log}")
    model.eval()
    return model

def transform_image(image) -> torch.Tensor:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_transformed, _ = transform(image, None)
    return image_transformed

class LangSAM():
    def __init__(self, sam_type="vit_h"):
        self.sam_type = sam_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_groundingdino()
        self.build_sam(sam_type)

    def build_sam(self, sam_type):
        checkpoint_url = SAM_MODELS[sam_type]
        try:
            sam = sam_model_registry[sam_type]()
            state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
            sam.load_state_dict(state_dict, strict=True)
        except:
            raise ValueError(f"Problem loading SAM please make sure you have the right model type: {sam_type} \
                and a working checkpoint: {checkpoint_url}. Recommend deleting the checkpoint and \
                re-downloading it.")
        sam.to(device=self.device)
        self.sam = SamPredictor(sam)

    def build_groundingdino(self):
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)

    def predict_dino(self, image_pil, text_prompt, box_threshold, text_threshold):
        image_trans = transform_image(image_pil)
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         device=self.device)
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        return boxes, logits, phrases

    def predict_sam(self, image_pil, boxes):
        image_array = np.asarray(image_pil)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        return masks.cpu()

    def predict(self, image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25):
        boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)
        return masks, boxes, phrases, logits


class Images:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = Image.open(image_path)

    @property
    def show(self):
        # Cargando la imagen
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10,10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def generation_masks_prompt(self, text_prompt):
        # Inicializa la clase LangSAM
        langsam = LangSAM(sam_type="vit_h")

        # Predicciones según texto de solicitud
        masks, boxes, phrases, logits = langsam.predict(self.image, text_prompt)

        return masks, boxes, phrases, logits

    def display_masks(self, masks, boxes, box=False):
        # Print the number of recognized masks
        print(f"Number of recognized masks: {len(masks)}")

        # Convertir imagen PIL a array de numpy
        image = np.array(self.image)

        # Crear una figura y ejes
        fig, ax = plt.subplots(1, figsize=(10, 10))

        if box:
            # Mostrar la imagen
            ax.imshow(image)

            # Recorrer cada caja y máscara
            for i, (box, mask) in enumerate(zip(boxes, masks)):
                # Dibujar un rectángulo rojo alrededor del objeto
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

                # Anotar el índice de la máscara en la esquina superior izquierda de la caja
                ax.text(x1, y1, str(i), color='r')

                # Sobreponer la máscara al objeto con cierta transparencia
                ax.imshow(np.where(mask > 0, 1, np.nan), alpha=0.6, cmap='Reds')
        else:
            # Crear una imagen en blanco del mismo tamaño que la imagen original
            mask_image = np.zeros_like(image)

            # Recorrer cada máscara
            for mask in masks:
                # Dibujar la máscara en la imagen en blanco
                for j in range(3):
                    mask_image[:,:,j] = np.where(mask, image[:,:,j], mask_image[:,:,j])

            # Mostrar la imagen de la máscara
            ax.imshow(mask_image)

        # Eliminar los ejes
        plt.axis('off')

        plt.show()

    def display_mask(self, masks, mask_index):
        # Lectura de la imagen
        image = PILImage.open(self.image_path)

        # Convertir imagen PIL a array de numpy
        image = np.array(image)

        # Crear una figura
        plt.figure(figsize=(10,10))

        # Mostrar la imagen
        plt.imshow(image)

        # Sobreponer la máscara al objeto con cierta transparencia
        plt.imshow(masks[mask_index], alpha=0.6)

        # Eliminar los ejes
        plt.axis('off')

        # Mostrar la imagen con la máscara
        plt.show()

    def display_reference_mask(self, text_prompt):
        # Predicciones según texto de solicitud
        masks, boxes, phrases, logits = self.generation_masks_prompt(text_prompt)

        # Muestra las máscaras usando la función display_masks
        self.display_masks(masks, boxes, box=True)



    def reference_mask(self, text_prompt, mask_index, matrix):
        # Generate the masks using the generation_masks_prompt function
        masks, _, _, _ = self.generation_masks_prompt(text_prompt)

        # Check if the mask_index is valid
        if mask_index >= len(masks):
            print(f"Invalid mask index. Only {len(masks)} masks available.")
            return None

        # Get the mask at the specified index
        mask = masks[mask_index]

        # If matrix is True, return the mask as a numpy array
        if matrix:
            return mask.numpy()

        # Otherwise, display the mask using the display_mask function
        else:
            self.display_mask(masks, mask_index)
            return None

    def normalize_masks(self, reference_mask, target_mask_index, masks, matrix: bool):
        # Load the original image
        image = Image.open(self.image_path)

        # Convert image PIL to numpy array
        image = np.array(image)

        # Create empty images of the same size as the original
        ref_image = np.zeros_like(image)
        target_image = np.zeros_like(image)

        # Apply the selected masks to the images
        for i in range(3):  # Loop over each channel
            ref_image[reference_mask == 1, i] = image[reference_mask == 1, i]
            target_image[masks[target_mask_index] == 1, i] = image[masks[target_mask_index] == 1, i]

        # Calculate the average color in the reference image for each channel
        avg_ref = np.zeros(3)
        for i in range(3):
            avg_ref[i] = np.mean(ref_image[ref_image[:, :, i] > 0, i])

        # Normalize the colors in the target image for each channel
        normalized_image = np.zeros_like(target_image, dtype=np.float32)
        for i in range(3):
            normalized_image[target_image[:, :, i] > 0, i] = (target_image[target_image[:, :, i] > 0, i] / avg_ref[i]) * 255.0

        # Ensure the values are within the range [0, 255]
        normalized_image = np.clip(normalized_image, 0, 255)

        if matrix:
            return normalized_image
        else:
            # Convert normalized image back to PIL Image for display
            normalized_image = Image.fromarray(normalized_image.astype(np.uint8))

            # Calculate the aspect ratio of the image
            aspect_ratio = image.shape[1] / image.shape[0]

            # Set the figure width
            figure_width = 10

            # Calculate the figure height based on the aspect ratio
            figure_height = figure_width / aspect_ratio

            # Set the figure size
            plt.figure(figsize=(figure_width, figure_height))

            # Display the image
            plt.imshow(normalized_image)
            plt.axis('off')

            return plt.show()

    def RGB_mask(self, target_mask_index, reference_mask_matrix, masks):
        # Generate the normalized mask
        normalized_mask = self.normalize_masks(reference_mask_matrix, target_mask_index, masks, matrix=True)

        # Calculate the average RGB values
        avg_R = np.mean(normalized_mask[normalized_mask[:,:,0] > 0, 0]) # Average of Red
        avg_G = np.mean(normalized_mask[normalized_mask[:,:,1] > 0, 1]) # Average of Green
        avg_B = np.mean(normalized_mask[normalized_mask[:,:,2] > 0, 2]) # Average of Blue

        return [avg_R, avg_G, avg_B]

    def generate_RGB_dataframe(self, reference_mask_matrix, masks):
        # Inicializar una lista para los datos
        data = []

        # Recorrer cada máscara
        for i in range(len(masks)):
            # Calculate the average RGB values of the current mask
            avg_RGB = self.RGB_mask(i, reference_mask_matrix, masks)

            # Append the index and average RGB values to the list
            data.append([i] + avg_RGB)

        # Convert the list to a DataFrame
        df = pd.DataFrame(data, columns=['Mask', 'R', 'G', 'B'])

        return df
        
    def rgb2xyz_custom(self, target_mask_index, reference_mask_matrix, masks, xyz_from_rgb=None):
        # Calculate average RGB values using RGB_mask function
        avg_RGB = self.RGB_mask(target_mask_index, reference_mask_matrix, masks)

        # Convert the average RGB values to the range [0,1]
        avg_RGB = [value / 255 for value in avg_RGB]

        # Prepare the RGB array for the conversion
        arr = _prepare_colorarray(avg_RGB).copy()

        # Apply the gamma correction
        mask = arr > 0.04045
        arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
        arr[~mask] /= 12.92

        # If no custom matrix is provided, use the standard sRGB to XYZ conversion matrix
        if xyz_from_rgb is None:
            xyz_from_rgb = np.array([
                [0.4124, 0.3576, 0.1805],
                [0.2126, 0.7152, 0.0722],
                [0.0193, 0.1192, 0.9505]
            ])

        # Perform the conversion
        xyz = arr @ xyz_from_rgb.T.astype(arr.dtype)

        # Multiply the XYZ values by 100 to scale them to the XYZ color space
        xyz *= 100

        return xyz

    def rgb2lab_custom(self, target_mask_index, reference_mask_matrix, masks, xyz_from_rgb=None):
        """
        Convert an RGB image to a LAB image.
        """
        # Convert from RGB to XYZ
        xyz = self.rgb2xyz_custom(target_mask_index, reference_mask_matrix, masks, xyz_from_rgb)
        
        # Normalize XYZ values to the range [0,1]
        xyz /= 100

        # Convert from XYZ to LAB
        lab = xyz2lab(xyz)

        return list(lab)

    def generate_LABCH_dataframe(self, reference_mask_matrix, masks, xyz_from_rgb=None):
        # Initialize a list to store the LAB values
        LAB_values = []

        # Iterate over the masks
        for i in range(len(masks)):
            # Calculate the LAB values of the current mask
            LAB = self.rgb2lab_custom(i, reference_mask_matrix, masks, xyz_from_rgb)

            # Append the LAB values to the list
            LAB_values.append(LAB)

        # Convert the list of LAB values into a pandas DataFrame
        df = pd.DataFrame(LAB_values, columns=['L', 'a', 'b'])

        # Calculate C and H values and add them as new columns to the dataframe
        df['C'] = sqrt(df['a']**2 + df['b']**2)
        df['H'] = degrees(arctan2(df['b'], df['a']))
        df.loc[df['H'] < 0, 'H'] += 360  # correct negative H values

        # Insert a new column for the mask indices
        df.insert(0, 'Mask', range(len(masks)))

        # Save the DataFrame as an Excel file
        df.to_excel("LABCH_masks.xlsx", index=False)

        # Return the DataFrame
        return df

    def calculate_mask_areas(self, masks, sort=False):
        # List to store mask information
        mask_info = []

        # Loop over each mask
        for i, mask in enumerate(masks):
            # Calculate the area of the mask (number of pixels)
            area = torch.sum(mask).item()

            # Add mask information to the list
            mask_info.append({
                'Mask': i,  # This will store the index of the mask
                'Area': area
            })

        # Create a DataFrame from the list of mask information
        mask_df = pd.DataFrame(mask_info)

        # If sort is True, sort the DataFrame by the area of the masks
        if sort:
            mask_df = mask_df.sort_values(by='Area')

        # Reset the index (the mask order)
        mask_df = mask_df.reset_index(drop=True)

        # Save the DataFrame to an Excel file
        mask_df.to_excel("area_masks.xlsx", index=False)

        return mask_df

    def plants_summary(self, reference_mask_matrix, masks, xyz_from_rgb=None, name=None):
        # Get the filename from the image_path

        if name != None:
            filename = name
        else:
            filename = os.path.basename(self.image_path)
            filename = filename[:-4]

        # Generate the RGB dataframe
        rgb_df = self.generate_RGB_dataframe(reference_mask_matrix, masks)

        # Generate the LABCH dataframe
        labch_df = self.generate_LABCH_dataframe(reference_mask_matrix, masks, xyz_from_rgb)

        # Calculate the mask areas
        area_df = self.calculate_mask_areas(masks)

        # Merge the dataframes on the 'Mask' column
        df = pd.merge(rgb_df, labch_df, on='Mask')
        df = pd.merge(df, area_df, on='Mask')

        # Insert a new column at the beginning for the filename
        df.insert(0, 'Filename', filename)

        # Save the DataFrame as an Excel file
        df.to_excel(f"{filename}.xlsx", index=False)

        # Return the DataFrame
        return df