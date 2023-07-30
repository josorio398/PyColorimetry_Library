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
from skimage import color
from skimage.color.colorconv import _prepare_colorarray

from numpy import sqrt, arctan2, degrees
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

from scipy.spatial.distance import cdist

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

    def normalize_masks(self, reference_mask, target_mask_index, masks, matrix):
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


    def generate_XYZ_dataframe(self, reference_mask_matrix, masks, xyz_from_rgb=None):
        # Initialize a list to store the XYZ values
        XYZ_values = []

        # Iterate over the masks
        for i in range(len(masks)):
            # Calculate the XYZ values of the current mask
            XYZ = self.rgb2xyz_custom(i, reference_mask_matrix, masks, xyz_from_rgb)

            # Append the XYZ values to the list
            XYZ_values.append([i] + list(XYZ))

        # Convert the list of XYZ values into a pandas DataFrame
        df = pd.DataFrame(XYZ_values, columns=['Mask', 'X', 'Y', 'Z'])

        # Save the DataFrame as an Excel file
        df.to_excel("XYZ_masks.xlsx", index=False)

        # Return the DataFrame
        return df


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

    def pantone_summary(self, theoretical_csv_path, mask_order, reference_mask_matrix, masks, reference_mask=None, xyz_from_rgb=None):
        # Load the theoretical values from the CSV file
        theoretical_df = pd.read_csv(theoretical_csv_path)

        # Generate the experimental DataFrame
        rgb_df = self.generate_RGB_dataframe(reference_mask_matrix, masks)
        xyz_df = self.generate_XYZ_dataframe(reference_mask_matrix, masks, xyz_from_rgb)
        labch_df = self.generate_LABCH_dataframe(reference_mask_matrix, masks, xyz_from_rgb)

        # Rename the columns in rgb_df
        #rgb_df = rgb_df.rename(columns={'R': 'RE', 'G': 'GE', 'B': 'BE'})

        # Drop the 'Mask' column from xyz_df and labch_df
        xyz_df = xyz_df.drop(columns=['Mask'])
        labch_df = labch_df.drop(columns=['Mask'])

        # Combine the RGB, XYZ, and LABCH dataframes
        experimental_df = pd.concat([rgb_df, xyz_df, labch_df], axis=1)

        # Create a mask order DataFrame
        mask_order_df = pd.DataFrame({'Mask': mask_order, 'order': range(len(mask_order))})

        # Convert 'Mask' column to int in both DataFrames
        mask_order_df['Mask'] = mask_order_df['Mask'].astype(int)
        experimental_df['Mask'] = experimental_df['Mask'].astype(int)

        # Merge the experimental_df with mask_order_df
        experimental_df = pd.merge(experimental_df, mask_order_df, on='Mask')

        # Sort values by 'order' column and drop 'order' column
        experimental_df = experimental_df.sort_values('order').drop('order', axis=1)

        # Add the mask areas to the DataFrame
        mask_areas_df = self.calculate_mask_areas(masks)
        mask_areas_df['Mask'] = mask_areas_df['Mask'].astype(int)

        # Merge the combined_df with mask_areas_df
        experimental_df = pd.merge(experimental_df, mask_areas_df, on='Mask')

        # Concatenate the theoretical and experimental DataFrames
        combined_df = pd.concat([theoretical_df, experimental_df.reset_index(drop=True)], axis=1)

        # Calculate the percentage error between RT and R and add it as the ER column
        combined_df['ΔR'] = np.abs(combined_df['RT'] - combined_df['R'])

        # Calculate the percentage error between GT and G and add it as the EG column
        combined_df['ΔG'] = np.abs(combined_df['GT'] - combined_df['G'])

        # Calculate the percentage error between BT and B and add it as the EB column
        combined_df['ΔB'] = np.abs(combined_df['BT'] - combined_df['B'])

        combined_df['ΔL'] = np.abs(combined_df['LT'] - combined_df['L'])

        combined_df['Δa'] = np.abs(combined_df['aT'] - combined_df['a'])

        combined_df['Δb'] = np.abs(combined_df['bT'] - combined_df['b'])

        # Calculate the Euclidean distance
        combined_df['ΔE'] = np.sqrt(combined_df['ΔL']**2 + combined_df['Δa']**2 + combined_df['Δb']**2)


        if reference_mask is None:
            reference_mask = mask_order[0]

        # Compute distances between the reference mask and the other masks
        ref_Lab = experimental_df.loc[reference_mask, ['L', 'a', 'b']]
        experimental_df['distances'] = np.sqrt((experimental_df['L'] - ref_Lab['L'])**2 +
                                                (experimental_df['a'] - ref_Lab['a'])**2 +
                                                (experimental_df['b'] - ref_Lab['b'])**2)

        # Add the 'distances' column to the combined_df DataFrame
        combined_df = pd.concat([combined_df, experimental_df['distances'].reset_index(drop=True)], axis=1)


        # Calculate the MAE for each error column and save it in a dictionary
        means = combined_df[['ΔR', 'ΔG', 'ΔB', 'ΔL', 'Δa', 'Δb','ΔE']].mean()

        # Create a new DataFrame for the MEANS and VALUES
        mape_df = pd.DataFrame({'MAE': means.index, 'VALUES': means.values})

        # Assign 'MAE' and 'VALUES' as two new columns in the existing dataframe
        combined_df = combined_df.assign(MAE=np.nan, VALUES=np.nan)
        combined_df.loc[:len(means)-1, ['MAE', 'VALUES']] = mape_df.values

        # Save the DataFrame to an Excel file
        combined_df.to_excel("pantone_summary.xlsx", index=False)

        return combined_df

class Data:
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):
            self.data = pd.read_excel(data)
            for column in ['R', 'G', 'B', 'L', 'a', 'b', 'C', 'H', 'Area']:
                self.data[column] = self.data[column].apply(lambda x: float(str(x).replace(',', '.')))
        else:
            raise ValueError('Data should be a pandas DataFrame or a string path to an Excel file.')

    @property
    def PlotCIELab(self):
        first_column_name = self.data.columns[0]

        # Get the LAB colors directly from the DataFrame
        lab_colors = self.data[['L', 'a', 'b']].values
        rgb_colors = self.data[['R', 'G', 'B']].values / 255.0  # Normalize to [0, 1]

        # Prepare the mask indices as custom data
        customdata = self.data[[first_column_name, 'Mask', 'L', 'a', 'b', 'C', 'H']].values

        # Create the 3D plot
        fig = go.Figure()

        # Add the colors
        fig.add_trace(go.Scatter3d(
            x=lab_colors[:,1],
            y=lab_colors[:,2],
            z=lab_colors[:,0],
            customdata=customdata,  # Add the mask indices as custom data
            mode='markers',
            marker=dict(
                size=6,
                color=['rgb({},{},{})'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in rgb_colors],
                opacity=1.0
            ),
            hovertemplate="<b>Filename</b>: %{customdata[0]}<br><b>Mask Index</b>: %{customdata[1]}<br><b>L</b>: %{customdata[2]:.2f}<br><b>A</b>: %{customdata[3]:.2f}<br><b>B</b>: %{customdata[4]:.2f}<br><b>C</b>: %{customdata[5]:.2f}<br><b>H</b>: %{customdata[6]:.2f}<extra></extra>",
        ))

        # Add the cutting ellipses
        theta = np.linspace(0, 2*np.pi, 100)

        # AB plane centered at (0,0,50)
        x = 127 * np.cos(theta)
        y = 127 * np.sin(theta)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=50*np.ones_like(x), mode='lines', line=dict(color='black', width=2, dash='dash'),showlegend=False))

        # LB plane centered at (0,0,50)
        x = 127 * np.cos(theta)
        z = 50 + 50 * np.sin(theta)
        fig.add_trace(go.Scatter3d(x=x, y=0*np.ones_like(x), z=z, mode='lines', line=dict(color='black', width=2, dash='dash'),showlegend=False))

        # LA plane centered at (0,0,50)
        y = 127 * np.cos(theta)
        z = 50 + 50 * np.sin(theta)
        fig.add_trace(go.Scatter3d(x=0*np.ones_like(y), y=y, z=z, mode='lines', line=dict(color='black', width=2, dash='dash'),showlegend=False))

        # Create the AB plane as a 3D mesh
        t = np.linspace(0, 1, 100)  # Parameter for interpolation
        X, Y = np.meshgrid(np.linspace(-127, 127, 100), np.linspace(-127, 127, 100))  # Create a grid over the entire AB space
        Z = 50 * np.ones_like(X)
        fig.add_trace(go.Mesh3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), opacity=0.2, color='gray',hoverinfo='skip'))

        # Añade los segmentos de línea
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 127], z=[50, 50], mode='lines', line=dict(color='black', width=2),showlegend=False)) # segmento AB
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, -127], z=[50, 50], mode='lines', line=dict(color='black', width=2),showlegend=False)) # segmento AB
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[100, 50], mode='lines', line=dict(color='black', width=2),showlegend=False)) # segmento LB
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 50], mode='lines', line=dict(color='black', width=2),showlegend=False)) # segmento LB
        fig.add_trace(go.Scatter3d(x=[0, 127], y=[0, 0], z=[50, 50], mode='lines', line=dict(color='black', width=2),showlegend=False)) # segmento LA
        fig.add_trace(go.Scatter3d(x=[0, -127], y=[0, 0], z=[50, 50], mode='lines', line=dict(color='black', width=2),showlegend=False)) # segmento LA

        # Define los límites del gráfico y los nombres de los ejes
        fig.update_layout(
            autosize=False,
            width=1000,
            height=1000,
            scene=dict(
                xaxis=dict(range=[-128, 128], title='a'),
                yaxis=dict(range=[-128, 128], title='b'),
                zaxis=dict(range=[0, 100], title='L')
            ),
        )

        # Muestra el gráfico
        fig.show()


    @property
    def Plotab(self):
        first_column_name = self.data.columns[0]

        # Get the AB values directly from the DataFrame
        ab_values = self.data[['a', 'b']].values
        rgb_colors = self.data[['R', 'G', 'B']].values / 255.0  # Normalize to [0, 1]

        # Prepare the mask indices as custom data
        customdata = self.data[[first_column_name, 'Mask', 'L', 'a', 'b', 'C', 'H']].values

        # Create the 2D plot
        fig = go.Figure()

        # Add the colors
        fig.add_trace(go.Scatter(
            x=ab_values[:,0],
            y=ab_values[:,1],
            customdata=customdata,  # Add the mask indices as custom data
            mode='markers',
            marker=dict(
                size=6,
                color=['rgb({},{},{})'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in rgb_colors],
                opacity=1.0
            ),
            hovertemplate="<b>Filename</b>: %{customdata[0]}<br><b>Mask Index</b>: %{customdata[1]}<br><b>L</b>: %{customdata[2]:.2f}<br><b>A</b>: %{customdata[3]:.2f}<br><b>B</b>: %{customdata[4]:.2f}<br><b>C</b>: %{customdata[5]:.2f}<br><b>H</b>: %{customdata[6]:.2f}<extra></extra>",
        ))

        # Add the line segments
        fig.add_trace(go.Scatter(x=[0, 0], y=[-127, 127], mode='lines', line=dict(color='black', width=1)))  # Vertical segment
        fig.add_trace(go.Scatter(x=[-127, 127], y=[0, 0], mode='lines', line=dict(color='black', width=1)))  # Horizontal segment

        # Add lines from the center to the circumference every 10 degrees
        for theta in np.arange(0, 2 * np.pi, np.pi / 18):  # 2 * np.pi / 18 gives a 10 degree increment
            x_end = 127 * np.cos(theta)
            y_end = 127 * np.sin(theta)
            fig.add_trace(go.Scatter(x=[0, x_end], y=[0, y_end], mode='lines',
                                      line=dict(color='rgba(128, 128, 128, 0.5)', width=1, dash='dot'), showlegend=False))

        # Add a circumference
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=-127, y0=-127, x1=127, y1=127,
            line=dict(color="black", width=2, dash="dash"),
        )

        # Define the plot limits and the axis names
        fig.update_layout(
            xaxis=dict(range=[-128, 128], title='a'),
            yaxis=dict(range=[-128, 128], title='b'),
            autosize=False,
            width=800,
            height=800,
        )

        # Show the plot
        fig.show()

    @property
    def PlotaL(self):
        first_column_name = self.data.columns[0]

        lb_values = self.data[['a', 'L']].values
        rgb_colors = self.data[['R', 'G', 'B']].values / 255.0
        customdata = self.data[[first_column_name, 'Mask', 'L', 'a', 'b', 'C', 'H']].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lb_values[:,0],
            y=lb_values[:,1],
            customdata=customdata,
            mode='markers',
            marker=dict(
                size=6,
                color=['rgb({},{},{})'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in rgb_colors],
                opacity=1.0
            ),
            hovertemplate="<b>Filename</b>: %{customdata[0]}<br><b>Mask Index</b>: %{customdata[1]}<br><b>L</b>: %{customdata[2]:.2f}<br><b>A</b>: %{customdata[3]:.2f}<br><b>B</b>: %{customdata[4]:.2f}<br><b>C</b>: %{customdata[5]:.2f}<br><b>H</b>: %{customdata[6]:.2f}<extra></extra>",
        ))

        fig.add_trace(go.Scatter(x=[0, 0], y=[0, 100], mode='lines', line=dict(color='black', width=1)))
        fig.add_trace(go.Scatter(x=[-127, 127], y=[50, 50], mode='lines', line=dict(color='black', width=1)))

        # Add an ellipse
        theta = np.linspace(0, 2*np.pi, 100)
        a = 127
        b = 50
        x = a * np.cos(theta)
        y = b * np.sin(theta) + 50  # Centered at L = 50
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='black', width=2, dash='dash')))

        fig.update_layout(
            xaxis=dict(range=[-128, 128], title='a'),
            yaxis=dict(range=[0, 100], title='L'),
            autosize=False,
            width=800,
            height=800,
        )
        fig.show()

    @property
    def elbow_method(self):
        # Convert colors to a numpy array for clustering
        color_values = self.data[['L', 'a', 'b']].values

        # Replace NaN values with the column mean
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        color_values_imputed = imputer.fit_transform(color_values)

        # Calculate the optimal number of clusters using the elbow method
        distortions = []
        K = range(1, 11)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(color_values_imputed)
            distortions.append(sum(np.min(cdist(color_values_imputed, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / color_values_imputed.shape[0])

        # Calculate the second derivative (approximate) of the distortions
        second_derivative = [0] + [distortions[i] - 2*distortions[i-1] + distortions[i-2] for i in range(2, len(distortions))]

        # The optimal number of clusters is where the second derivative is maximum
        optimal_k = np.argmax(second_derivative) + 1

        print(f'Optimal number of clusters: {optimal_k}')

        # Create the plot
        fig = go.Figure(data=go.Scatter(x=list(K), y=distortions, mode='lines+markers'))

        # Customize aspect
        fig.update_layout(title='Elbow Method',
                          xaxis=dict(title='Number of clusters (k)'),
                          yaxis=dict(title='Distortion Score'))

        fig.show()

    def clusters(self, n_clusters):
        # Convert color values to a NumPy array for clustering
        color_values = self.data[['L', 'a', 'b']].values

        # Replace NaN values with the column mean
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        color_values_imputed = imputer.fit_transform(color_values)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(color_values_imputed)

        # Make a copy of the original DataFrame
        modified_data = self.data.copy()

        # Add the cluster labels to the new DataFrame
        modified_data['cluster'] = kmeans.labels_

        # Save the new DataFrame to an Excel file
        modified_data.to_excel('cluster_data.xlsx', index=False)

        # Return the new DataFrame
        return modified_data

    def PlotClusterCIELab(self, n_clusters, distinction=None):
        # First, we obtain the dataframe with the cluster assignment
        clustered_data = self.clusters(n_clusters)

        # Get the LAB colors directly from the DataFrame
        lab_colors = clustered_data[['L', 'a', 'b']].values

        # Prepare the custom data
        customdata = clustered_data[['Filename', 'Mask', 'L', 'a', 'b', 'C', 'H', 'cluster']].values

        # Create the 3D plot
        fig = go.Figure()

        # Define marker shapes
        marker_shapes = ['circle','cross','square','x','diamond']

        # Placeholder for all the traces
        traces = []

        # Add the colors
        for i in range(n_clusters):
            mask = customdata[:,7] == i
            color_to_use = []
            for filename, r, g, b in zip(customdata[mask, 0], clustered_data['R'][mask], clustered_data['G'][mask], clustered_data['B'][mask]):
                if distinction is not None and filename in distinction:
                    color_to_use.append(distinction[filename])
                else:
                    color_to_use.append('rgb({},{},{})'.format(int(r), int(g), int(b)))

            traces.append(go.Scatter3d(
                x=lab_colors[mask, 1],
                y=lab_colors[mask, 2],
                z=lab_colors[mask, 0],
                customdata=customdata[mask],  # Add the mask indices as custom data
                mode='markers',
                marker=dict(
                    size=6,
                    color=color_to_use,
                    opacity=1.0,
                    symbol=marker_shapes[i % len(marker_shapes)]  # Use a different marker shape for each cluster
                ),
                hovertemplate="<b>Filename</b>: %{customdata[0]}<br><b>Mask Index</b>: %{customdata[1]}<br><b>L</b>: %{customdata[2]:.2f}<br><b>A</b>: %{customdata[3]:.2f}<br><b>B</b>: %{customdata[4]:.2f}<br><b>C</b>: %{customdata[5]:.2f}<br><b>H</b>: %{customdata[6]:.2f}<br><b>Cluster</b>: %{customdata[7]}<extra></extra>",
            ))

        # Add the cutting ellipses
        theta = np.linspace(0, 2*np.pi, 100)

        # AB plane centered at (0,0,50)
        x = 127 * np.cos(theta)
        y = 127 * np.sin(theta)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=50*np.ones_like(x), mode='lines', line=dict(color='black', width=2, dash='dash'),showlegend=False))

        # LB plane centered at (0,0,50)
        x = 127 * np.cos(theta)
        z = 50 + 50 * np.sin(theta)
        fig.add_trace(go.Scatter3d(x=x, y=0*np.ones_like(x), z=z, mode='lines', line=dict(color='black', width=2, dash='dash'),showlegend=False))

        # LA plane centered at (0,0,50)
        y = 127 * np.cos(theta)
        z = 50 + 50 * np.sin(theta)
        fig.add_trace(go.Scatter3d(x=0*np.ones_like(y), y=y, z=z, mode='lines', line=dict(color='black', width=2, dash='dash'),showlegend=False))

        # Create the AB plane as a 3D mesh
        t = np.linspace(0, 1, 100)  # Parameter for interpolation
        X, Y = np.meshgrid(np.linspace(-127, 127, 100), np.linspace(-127, 127, 100))  # Create a grid over the entire AB space
        Z = 50 * np.ones_like(X)
        fig.add_trace(go.Mesh3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), opacity=0.2, color='gray',hoverinfo='skip'))

        # Add all the traces to the figure at the same time
        for trace in traces:
            fig.add_trace(trace)

        # Define los límites del gráfico y los nombres de los ejes
        fig.update_layout(
            autosize=False,
            width=1000,
            height=1000,
            scene=dict(
                xaxis=dict(range=[-128, 128], title='a'),
                yaxis=dict(range=[-128, 128], title='b'),
                zaxis=dict(range=[0, 100], title='L')
            ),
        )

        # Muestra el gráfico
        fig.show()

    def PlotClusterab(self, n_clusters, distinction=None):
        # First, we obtain the dataframe with the cluster assignment
        clustered_data = self.clusters(n_clusters)

        # Get the AB values directly from the DataFrame
        ab_values = clustered_data[['a', 'b']].values

        # Prepare the custom data
        customdata = clustered_data[['Filename', 'Mask', 'L', 'a', 'b', 'C', 'H', 'cluster']].values

        # Create the 2D plot
        fig = go.Figure()

        # Define marker shapes
        marker_shapes = ['circle', 'cross', 'square', 'x', 'diamond']

        # Add the colors
        for i in range(n_clusters):
            mask = customdata[:, 7] == i
            color_to_use = []
            for filename, r, g, b in zip(customdata[mask, 0], clustered_data['R'][mask], clustered_data['G'][mask], clustered_data['B'][mask]):
                if distinction is not None and filename in distinction:
                    color_to_use.append(distinction[filename])
                else:
                    color_to_use.append('rgb({},{},{})'.format(int(r), int(g), int(b)))

            fig.add_trace(go.Scatter(
                x=ab_values[mask, 0],
                y=ab_values[mask, 1],
                customdata=customdata[mask],  # Add the mask indices as custom data
                mode='markers',
                marker=dict(
                    size=6,
                    color=color_to_use,
                    opacity=1.0,
                    symbol=marker_shapes[i % len(marker_shapes)]  # Use a different marker shape for each cluster
                ),
                hovertemplate="<b>Filename</b>: %{customdata[0]}<br><b>Mask Index</b>: %{customdata[1]}<br><b>L</b>: %{customdata[2]:.2f}<br><b>A</b>: %{customdata[3]:.2f}<br><b>B</b>: %{customdata[4]:.2f}<br><b>C</b>: %{customdata[5]:.2f}<br><b>H</b>: %{customdata[6]:.2f}<br><b>Cluster</b>: %{customdata[7]}<extra></extra>",
            ))

        # Add the line segments
        fig.add_trace(go.Scatter(x=[0, 0], y=[-127, 127], mode='lines', line=dict(color='black', width=1)))  # Vertical segment
        fig.add_trace(go.Scatter(x=[-127, 127], y=[0, 0], mode='lines', line=dict(color='black', width=1)))  # Horizontal segment

        # Add lines from the center to the circumference every 10 degrees
        for theta in np.arange(0, 2 * np.pi, np.pi / 18):  # 2 * np.pi / 18 gives a 10 degree increment
            x_end = 127 * np.cos(theta)
            y_end = 127 * np.sin(theta)
            fig.add_trace(go.Scatter(x=[0, x_end], y=[0, y_end], mode='lines',
                                    line=dict(color='rgba(128, 128, 128, 0.5)', width=1, dash='dot'), showlegend=False))

        # Add a circumference
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=-127, y0=-127, x1=127, y1=127,
            line=dict(color="black", width=2, dash="dash"),
        )

        # Define the plot limits and the axis names
        fig.update_layout(
            xaxis=dict(range=[-128, 128], title='a'),
            yaxis=dict(range=[-128, 128], title='b'),
            autosize=False,
            width=800,
            height=800,
        )

        # Show the plot
        fig.show()


    def PlotClusteraL(data, n_clusters, distinction=None):
        # First, we obtain the dataframe with the cluster assignment
        clustered_data = data.clusters(n_clusters)

        # Get the aL values directly from the DataFrame
        aL_values = clustered_data[['a', 'L']].values

        # Prepare the custom data
        customdata = clustered_data[['Filename', 'Mask', 'L', 'a', 'b', 'C', 'H', 'cluster']].values

        # Create the 2D plot
        fig = go.Figure()

        # Define marker shapes
        marker_shapes = ['circle','cross','square','x','diamond']

        # Add the colors
        for i in range(n_clusters):
            mask = customdata[:,7] == i
            color_to_use = []
            for filename, r, g, b in zip(customdata[mask, 0], clustered_data['R'][mask], clustered_data['G'][mask], clustered_data['B'][mask]):
                if distinction is not None and filename in distinction:
                    color_to_use.append(distinction[filename])
                else:
                    color_to_use.append('rgb({},{},{})'.format(int(r), int(g), int(b)))

            fig.add_trace(go.Scatter(
                x=aL_values[mask,0],
                y=aL_values[mask,1],
                customdata=customdata[mask],  # Add the mask indices as custom data
                mode='markers',
                marker=dict(
                    size=6,
                    color=color_to_use,
                    opacity=1.0,
                    symbol=marker_shapes[i % len(marker_shapes)]  # Use a different marker shape for each cluster
                ),
                hovertemplate="<b>Filename</b>: %{customdata[0]}<br><b>Mask Index</b>: %{customdata[1]}<br><b>L</b>: %{customdata[2]:.2f}<br><b>A</b>: %{customdata[3]:.2f}<br><b>B</b>: %{customdata[4]:.2f}<br><b>C</b>: %{customdata[5]:.2f}<br><b>H</b>: %{customdata[6]:.2f}<br><b>Cluster</b>: %{customdata[7]}<extra></extra>",
            ))

        # Add line segments
        fig.add_trace(go.Scatter(x=[0, 0], y=[0, 100], mode='lines', line=dict(color='black', width=1)))  # Vertical segment
        fig.add_trace(go.Scatter(x=[-127, 127], y=[50, 50], mode='lines', line=dict(color='black', width=1)))  # Horizontal segment

        # Add an ellipse
        theta = np.linspace(0, 2*np.pi, 100)
        a = 127
        b = 50
        x = a * np.cos(theta)
        y = b * np.sin(theta) + 50  # Centered at L = 50
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='black', width=2, dash='dash')))

        # Define the plot limits and the axis names
        fig.update_layout(
            xaxis=dict(range=[-128, 128], title='a'),
            yaxis=dict(range=[0, 100], title='L'),
            autosize=False,
            width=800,
            height=800,
        )

        # Show the plot
        fig.show()

