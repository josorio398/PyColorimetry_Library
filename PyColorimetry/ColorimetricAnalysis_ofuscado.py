import cv2
import os
import groundingdino . datasets . transforms as T
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL import Image as PILImage
import matplotlib . pyplot as plt
import matplotlib . patches as patches
if 82 - 82: Iii1i
from groundingdino . models import build_model
from groundingdino . util import box_ops
from groundingdino . util . inference import predict
from groundingdino . util . slconfig import SLConfig
from groundingdino . util . utils import clean_state_dict
if 87 - 87: Ii % i1i1i1111I . Oo / OooOoo * I1Ii1I1 - I1I
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from segment_anything import SamAutomaticMaskGenerator
if 81 - 81: i1 + ooOOO / oOo0O00 * i1iiIII111 * IiIIii11Ii
from skimage . color import rgb2xyz
from skimage . color import xyz2lab
from skimage import color
from skimage . color . colorconv import _prepare_colorarray
if 84 - 84: ooo000 - Ooo0Ooo + iI1iII1I1I1i . IIiIIiIi11I1
from numpy import sqrt , arctan2 , degrees
import plotly . graph_objects as go
if 98 - 98: I11iiIi11i1I % oOO
from sklearn . cluster import KMeans
from sklearn . impute import SimpleImputer
if 31 - 31: i1I
from scipy . spatial . distance import cdist
if 9 - 9: IiI11Ii111 / oOo0O00 / IiIIii11Ii - I11iiIi11i1I - iI1iII1I1I1i
import warnings
if 16 - 16: i1i1i1111I / i1iiIII111
# Ignorar las advertencias de categoría FutureWarning y UserWarning
warnings . filterwarnings ( 'ignore' , category = FutureWarning )
warnings . filterwarnings ( 'ignore' , category = UserWarning )
if 3 - 3: i1 % i1 % i1i1i1111I . Ii * i1
if 9 - 9: i1iiIII111
i1Ii1i = {
 "vit_h" : "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
 }
if 93 - 93: IIiIIiIi11I1 % IIiIIiIi11I1 / I1I - Oo . Ooo0Ooo
OO0o000o = os . environ . get ( "TORCH_HOME" , os . path . expanduser ( "~/.cache/torch/hub/checkpoints" ) )
if 8 - 8: i1iiIII111 * i1 . I1I / oOO
def ooOOooO0 ( repo_id , filename , ckpt_config_filename , device = 'gpu' ) :
 i1iiiiIIIiIi = hf_hub_download ( repo_id = repo_id , filename = ckpt_config_filename )
 II = SLConfig . fromfile ( i1iiiiIIIiIi )
 OO0000 = build_model ( II )
 II . device = device
 oOoo0 = hf_hub_download ( repo_id = repo_id , filename = filename )
 Iio0 = torch . load ( oOoo0 , map_location = 'cpu' )
 i1i = OO0000 . load_state_dict ( clean_state_dict ( Iio0 [ 'model' ] ) , strict = False )
 if 87 - 87: Oo - i1I
 OO0000 . eval ( )
 return OO0000
 if 32 - 32: Ii % i1i1i1111I % i1I - I11iiIi11i1I % i1iiIII111
def o0OoOOo00OOO ( image ) -> torch . Tensor :
 i11ii1i1I = T . Compose ( [
 T . RandomResize ( [ 800 ] , max_size = 1333 ) ,
 T . ToTensor ( ) ,
 T . Normalize ( [ 0.485 , 0.456 , 0.406 ] , [ 0.229 , 0.224 , 0.225 ] ) ,
 ] )
 IiI , o00O000o0 = i11ii1i1I ( image , None )
 return IiI
 if 71 - 71: Oo * ooOOO / Oo
class iI1i1I ( ) :
 def __init__ ( self , sam_type = "vit_h" ) :
  self . sam_type = sam_type
  self . device = torch . device ( "cuda" if torch . cuda . is_available ( ) else "cpu" )
  self . build_groundingdino ( )
  self . build_sam ( sam_type )
  if 27 - 27: ooOOO * IIiIIiIi11I1 % Ii + Ooo0Ooo . Ii + Iii1i
 def build_sam ( self , sam_type ) :
  oOo0O00O0ooo = i1Ii1i [ sam_type ]
  try :
   i11iIii = sam_model_registry [ sam_type ] ( )
   OO = torch . hub . load_state_dict_from_url ( oOo0O00O0ooo )
   i11iIii . load_state_dict ( OO , strict = True )
  except :
   raise ValueError ( "Problem loading SAM please make sure you have the right model type: {0} \
                and a working checkpoint: {1}. Recommend deleting the checkpoint and \
                re-downloading it." . format ( sam_type , oOo0O00O0ooo ) )
  i11iIii . to ( device = self . device )
  self . sam = SamPredictor ( i11iIii )
  if 27 - 27: Oo / oOO + IiI11Ii111 - OooOoo * I1Ii1I1 / I1Ii1I1
 def build_groundingdino ( self ) :
  o0OO = "ShilongLiu/GroundingDINO"
  I1iI = "groundingdino_swinb_cogcoor.pth"
  IiIII = "GroundingDINO_SwinB.cfg.py"
  self . groundingdino = ooOOooO0 ( o0OO , I1iI , IiIII )
  if 30 - 30: I1I . ooo000
 def predict_dino ( self , image_pil , text_prompt , box_threshold , text_threshold ) :
  OOO = o0OoOOo00OOO ( image_pil )
  OooOoo0OO0OO0 , IiIi1Ii1111 , Ii1I1I1i = predict ( model = self . groundingdino ,
 image = OOO ,
 caption = text_prompt ,
 box_threshold = box_threshold ,
 text_threshold = text_threshold ,
 device = self . device )
  OOOO0O0ooO0O , I1iIIiI1 = image_pil . size
  OooOoo0OO0OO0 = box_ops . box_cxcywh_to_xyxy ( OooOoo0OO0OO0 ) * torch . Tensor ( [ OOOO0O0ooO0O , I1iIIiI1 , OOOO0O0ooO0O , I1iIIiI1 ] )
  return OooOoo0OO0OO0 , IiIi1Ii1111 , Ii1I1I1i
  if 78 - 78: iI1iII1I1I1i + iI1iII1I1I1i - I1I * IIiIIiIi11I1 % I1Ii1I1 * i1i1i1111I
 def predict_sam ( self , image_pil , boxes ) :
  Oo0 = np . asarray ( image_pil )
  self . sam . set_image ( Oo0 )
  ooooo = self . sam . transform . apply_boxes_torch ( boxes , Oo0 . shape [ : 2 ] )
  oO00o00OO , o00O000o0 , o00O000o0 = self . sam . predict_torch (
 point_coords = None ,
 point_labels = None ,
 boxes = ooooo . to ( self . sam . device ) ,
 multimask_output = False ,
 )
  return oO00o00OO . cpu ( )
  if 52 - 52: Iii1i / Oo
 def predict ( self , image_pil , text_prompt , box_threshold = 0.3 , text_threshold = 0.25 ) :
  OooOoo0OO0OO0 , IiIi1Ii1111 , Ii1I1I1i = self . predict_dino ( image_pil , text_prompt , box_threshold , text_threshold )
  oO00o00OO = torch . tensor ( [ ] )
  if len ( OooOoo0OO0OO0 ) > 0 :
   oO00o00OO = self . predict_sam ( image_pil , OooOoo0OO0OO0 )
   oO00o00OO = oO00o00OO . squeeze ( 1 )
  return oO00o00OO , OooOoo0OO0OO0 , Ii1I1I1i , IiIi1Ii1111
  if 100 - 100: I1Ii1I1 / I11iiIi11i1I * Ii - oOO
  if 32 - 32: iI1iII1I1I1i
class I111II111I1I :
 def __init__ ( self , image_path ) :
  self . image_path = image_path
  self . image = Image . open ( image_path )
  if 21 - 21: iI1iII1I1I1i - i1 + ooOOO * IIiIIiIi11I1 % i1 * ooOOO
 @ property
 def show ( self ) :
  if 57 - 57: oOo0O00
  II1iiii = cv2 . imread ( self . image_path )
  II1iiii = cv2 . cvtColor ( II1iiii , cv2 . COLOR_BGR2RGB )
  if 50 - 50: I1I / Ooo0Ooo . i1iiIII111 - iI1iII1I1I1i % Ii - I1I
  plt . figure ( figsize = ( 10 , 10 ) )
  plt . imshow ( II1iiii )
  plt . axis ( 'off' )
  plt . show ( )
  if 54 - 54: Iii1i % ooo000 % Iii1i - IiIIii11Ii
 def generation_masks_prompt ( self , text_prompt ) :
  if 39 - 39: oOO - oOO * i1 % IIiIIiIi11I1
  IIiII11I = iI1i1I ( sam_type = "vit_h" )
  if 93 - 93: OooOoo - IiI11Ii111
  if 69 - 69: ooo000
  oO00o00OO , OooOoo0OO0OO0 , Ii1I1I1i , IiIi1Ii1111 = IIiII11I . predict ( self . image , text_prompt )
  if 76 - 76: Iii1i * Ooo0Ooo . iI1iII1I1I1i / Ii / ooOOO
  return oO00o00OO , OooOoo0OO0OO0 , Ii1I1I1i , IiIi1Ii1111
  if 49 - 49: iI1iII1I1I1i / i1iiIII111 + ooo000
 def display_masks ( self , masks , boxes , box = False ) :
  if 36 - 36: i1i1i1111I + Iii1i - oOO * Ii
  print ( "Number of recognized masks: {0}" . format ( len ( masks ) ) )
  if 45 - 45: i1i1i1111I * Ii
  if 97 - 97: i1
  II1iiii = np . array ( self . image )
  if 26 - 26: IiI11Ii111
  if 20 - 20: IIiIIiIi11I1 / Oo
  IIIi1111iiIi1 , I1II1ii111i = plt . subplots ( 1 , figsize = ( 10 , 10 ) )
  if 14 - 14: IiI11Ii111 + i1I . IiIIii11Ii . Ooo0Ooo % IiIIii11Ii * i1i1i1111I
  if box :
   if 65 - 65: Iii1i + IIiIIiIi11I1 - Ooo0Ooo . iI1iII1I1I1i + OooOoo * Ooo0Ooo
   I1II1ii111i . imshow ( II1iiii )
   if 23 - 23: ooOOO % oOO % iI1iII1I1I1i - i1I - i1iiIII111 + i1
   if 12 - 12: i1 - i1I - IiI11Ii111
   for OOooOOOO00O , ( box , ii1 ) in enumerate ( zip ( boxes , masks ) ) :
    if 53 - 53: i1 % IIiIIiIi11I1 * IIiIIiIi11I1 * iI1iII1I1I1i
    iiI1iI , O0OO00OO0O , Ooo0oO , ii1II = box
    oOoOo = patches . Rectangle ( ( iiI1iI , O0OO00OO0O ) , Ooo0oO - iiI1iI , ii1II - O0OO00OO0O , linewidth = 1 , edgecolor = 'r' , facecolor = 'none' )
    I1II1ii111i . add_patch ( oOoOo )
    if 39 - 39: oOO
    if 17 - 17: Ii . oOo0O00 % OooOoo
    I1II1ii111i . text ( iiI1iI , O0OO00OO0O , str ( OOooOOOO00O ) , color = 'r' )
    if 82 - 82: Iii1i . oOo0O00 % IIiIIiIi11I1 - iI1iII1I1I1i
    if 44 - 44: ooOOO . IIiIIiIi11I1
    I1II1ii111i . imshow ( np . where ( ii1 > 0 , 1 , np . nan ) , alpha = 0.6 , cmap = 'Reds' )
  else :
   if 15 - 15: ooo000
   o00ooo0Oooo = np . zeros_like ( II1iiii )
   if 72 - 72: IiI11Ii111 % i1 % ooo000 % iI1iII1I1I1i
   if 57 - 57: Ooo0Ooo % i1I + IiI11Ii111
   for ii1 in masks :
    if 8 - 8: Oo * I1I / i1i1i1111I + i1 / I1Ii1I1
    for iI in range ( 3 ) :
     o00ooo0Oooo [ : , : , iI ] = np . where ( ii1 , II1iiii [ : , : , iI ] , o00ooo0Oooo [ : , : , iI ] )
     if 58 - 58: IIiIIiIi11I1 % Ii + oOo0O00 / oOo0O00 * iI1iII1I1I1i * i1i1i1111I
     if 88 - 88: i1I + I1I
   I1II1ii111i . imshow ( o00ooo0Oooo )
   if 55 - 55: oOO
   if 29 - 29: ooOOO / i1i1i1111I / I1Ii1I1 . IiI11Ii111 + i1I * oOo0O00
  plt . axis ( 'off' )
  if 57 - 57: iI1iII1I1I1i
  plt . show ( )
  if 89 - 89: ooo000 * I11iiIi11i1I + oOO
 def display_mask ( self , masks , mask_index ) :
  if 64 - 64: IiI11Ii111 . Ooo0Ooo . i1iiIII111 * IiI11Ii111
  II1iiii = PILImage . open ( self . image_path )
  if 33 - 33: i1i1i1111I
  if 30 - 30: i1I % ooOOO . Ii % IIiIIiIi11I1 / iI1iII1I1I1i % ooOOO
  II1iiii = np . array ( II1iiii )
  if 24 - 24: IIiIIiIi11I1 - IIiIIiIi11I1 . Ooo0Ooo + i1iiIII111 + Oo
  if 21 - 21: IIiIIiIi11I1 - ooo000 + oOO
  plt . figure ( figsize = ( 10 , 10 ) )
  if 5 - 5: i1iiIII111 . i1iiIII111 + ooo000 . I11iiIi11i1I
  if 1 - 1: iI1iII1I1I1i % Ii - I11iiIi11i1I / IiI11Ii111 + iI1iII1I1I1i - Ii
  plt . imshow ( II1iiii )
  if 27 - 27: OooOoo % iI1iII1I1I1i + IIiIIiIi11I1
  if 16 - 16: IiI11Ii111
  plt . imshow ( masks [ mask_index ] , alpha = 0.6 )
  if 31 - 31: oOo0O00 / Iii1i % Ooo0Ooo % i1 . iI1iII1I1I1i . Oo
  if 83 - 83: oOo0O00 - I11iiIi11i1I
  plt . axis ( 'off' )
  if 91 - 91: IIiIIiIi11I1 - i1 - iI1iII1I1I1i
  if 71 - 71: I1I - IiI11Ii111
  plt . show ( )
  if 66 - 66: i1i1i1111I / Ooo0Ooo + I11iiIi11i1I + Iii1i + oOo0O00 + i1I
 def display_reference_mask ( self , text_prompt ) :
  if 75 - 75: OooOoo - oOO - IiIIii11Ii - oOO + ooo000 % iI1iII1I1I1i
  oO00o00OO , OooOoo0OO0OO0 , Ii1I1I1i , IiIi1Ii1111 = self . generation_masks_prompt ( text_prompt )
  if 42 - 42: i1 * i1I
  if 50 - 50: Ii - i1iiIII111
  self . display_masks ( oO00o00OO , OooOoo0OO0OO0 , box = True )
  if 96 - 96: ooo000 * OooOoo - Ii - OooOoo
  if 65 - 65: Oo + Oo - iI1iII1I1I1i % OooOoo . Ooo0Ooo
  if 84 - 84: IIiIIiIi11I1 . ooOOO
 def reference_mask ( self , text_prompt , mask_index , matrix ) :
  if 44 - 44: ooo000 * i1i1i1111I * oOO + i1iiIII111 - IIiIIiIi11I1
  oO00o00OO , o00O000o0 , o00O000o0 , o00O000o0 = self . generation_masks_prompt ( text_prompt )
  if 70 - 70: IiI11Ii111
  if 9 - 9: oOo0O00 * i1
  if mask_index >= len ( oO00o00OO ) :
   print ( "Invalid mask index. Only {0} masks available." . format ( len ( oO00o00OO ) ) )
   return None
   if 96 - 96: Ooo0Ooo
   if 13 - 13: Oo * I1Ii1I1 - oOo0O00 * Ii . Ii + oOo0O00
  ii1 = oO00o00OO [ mask_index ]
  if 46 - 46: OooOoo - I11iiIi11i1I / Ooo0Ooo
  if 73 - 73: I1Ii1I1 / i1i1i1111I / ooo000 % i1 % i1I - OooOoo
  if matrix :
   return ii1 . numpy ( )
   if 30 - 30: ooOOO * ooOOO - Iii1i * iI1iII1I1I1i
   if 37 - 37: I1Ii1I1 % iI1iII1I1I1i . i1I + Ooo0Ooo + ooOOO * iI1iII1I1I1i
  else :
   self . display_mask ( oO00o00OO , mask_index )
   return None
   if 39 - 39: IIiIIiIi11I1 - Oo
 def normalize_masks ( self , reference_mask , target_mask_index , masks , matrix ) :
  if 31 - 31: IiIIii11Ii % oOo0O00 % oOo0O00 * Iii1i
  II1iiii = Image . open ( self . image_path )
  if 85 - 85: Iii1i + Ii % IIiIIiIi11I1 % oOo0O00
  if 100 - 100: IiIIii11Ii % i1
  II1iiii = np . array ( II1iiii )
  if 82 - 82: ooOOO % OooOoo
  if 81 - 81: Ii
  O0 = np . zeros_like ( II1iiii )
  oOOoo0O0 = np . zeros_like ( II1iiii )
  if 65 - 65: ooOOO / oOo0O00 - i1iiIII111
  if 15 - 15: OooOoo . ooo000 / IiIIii11Ii % i1i1i1111I
  for OOooOOOO00O in range ( 3 ) :
   O0 [ reference_mask == 1 , OOooOOOO00O ] = II1iiii [ reference_mask == 1 , OOooOOOO00O ]
   oOOoo0O0 [ masks [ target_mask_index ] == 1 , OOooOOOO00O ] = II1iiii [ masks [ target_mask_index ] == 1 , OOooOOOO00O ]
   if 51 - 51: ooOOO
   if 69 - 69: iI1iII1I1I1i
  OO0Oo0oOo00 = np . zeros ( 3 )
  for OOooOOOO00O in range ( 3 ) :
   OO0Oo0oOo00 [ OOooOOOO00O ] = np . mean ( O0 [ O0 [ : , : , OOooOOOO00O ] > 0 , OOooOOOO00O ] )
   if 42 - 42: i1i1i1111I / iI1iII1I1I1i * oOo0O00 * Ooo0Ooo * Oo
   if 18 - 18: Ii
  Ooo = np . zeros_like ( oOOoo0O0 , dtype = np . float32 )
  for OOooOOOO00O in range ( 3 ) :
   Ooo [ oOOoo0O0 [ : , : , OOooOOOO00O ] > 0 , OOooOOOO00O ] = ( oOOoo0O0 [ oOOoo0O0 [ : , : , OOooOOOO00O ] > 0 , OOooOOOO00O ] / OO0Oo0oOo00 [ OOooOOOO00O ] ) * 255.0
   if 69 - 69: iI1iII1I1I1i - IiIIii11Ii
   if 86 - 86: oOO * i1I - OooOoo * Oo - IiI11Ii111 - Ooo0Ooo
  Ooo = np . clip ( Ooo , 0 , 255 )
  if 36 - 36: IiIIii11Ii
  if matrix :
   return Ooo
  else :
   if 28 - 28: i1 - Oo + IiIIii11Ii * oOO
   Ooo = Image . fromarray ( Ooo . astype ( np . uint8 ) )
   if 83 - 83: ooo000 . I1I + Oo
   if 59 - 59: I1I
   iii1ii1 = II1iiii . shape [ 1 ] / II1iiii . shape [ 0 ]
   if 27 - 27: i1 / OooOoo + Iii1i % OooOoo + OooOoo
   if 70 - 70: Oo / Iii1i * OooOoo
   I1Iii = 10
   if 34 - 34: OooOoo
   if 95 - 95: Oo % i1i1i1111I * iI1iII1I1I1i - IiIIii11Ii
   Ii1i111II1 = I1Iii / iii1ii1
   if 4 - 4: IIiIIiIi11I1
   if 52 - 52: Oo * iI1iII1I1I1i - i1i1i1111I . i1iiIII111
   plt . figure ( figsize = ( I1Iii , Ii1i111II1 ) )
   if 78 - 78: OooOoo . ooOOO
   if 80 - 80: oOO % i1iiIII111 * IiI11Ii111 - oOo0O00 % I11iiIi11i1I - IiI11Ii111
   plt . imshow ( Ooo )
   plt . axis ( 'off' )
   if 56 - 56: Oo
   return plt . show ( )
   if 84 - 84: I1I % iI1iII1I1I1i - Ooo0Ooo / iI1iII1I1I1i + Ooo0Ooo - Oo
 def RGB_mask ( self , target_mask_index , reference_mask_matrix , masks ) :
  if 41 - 41: ooOOO + OooOoo + IIiIIiIi11I1 * i1i1i1111I
  iIi = self . normalize_masks ( reference_mask_matrix , target_mask_index , masks , matrix = True )
  if 96 - 96: OooOoo + Ii % oOO % i1iiIII111 % Oo * Ooo0Ooo
  if 46 - 46: Ii - IiI11Ii111 + IiIIii11Ii + i1I . I1I % OooOoo
  oO0 = np . mean ( iIi [ iIi [ : , : , 0 ] > 0 , 0 ] )
  o0oO00 = np . mean ( iIi [ iIi [ : , : , 1 ] > 0 , 1 ] )
  IiOOooo00 = np . mean ( iIi [ iIi [ : , : , 2 ] > 0 , 2 ] )
  if 52 - 52: OooOoo
  return [ oO0 , o0oO00 , IiOOooo00 ]
  if 95 - 95: i1i1i1111I . I11iiIi11i1I * ooo000 / iI1iII1I1I1i - I1Ii1I1 + IiIIii11Ii
 def generate_RGB_dataframe ( self , reference_mask_matrix , masks ) :
  if 40 - 40: i1I + i1i1i1111I % OooOoo . IiI11Ii111 / i1i1i1111I . ooOOO
  O0OOOo0oO = [ ]
  if 56 - 56: oOo0O00
  if 87 - 87: IIiIIiIi11I1 % oOo0O00 % i1 . IiI11Ii111 - Iii1i . I1I
  for OOooOOOO00O in range ( len ( masks ) ) :
   if 39 - 39: ooOOO / oOo0O00
   oO = self . RGB_mask ( OOooOOOO00O , reference_mask_matrix , masks )
   if 96 - 96: OooOoo / oOO - i1 * I11iiIi11i1I
   if 72 - 72: i1i1i1111I + Ii - Iii1i - i1i1i1111I - i1I + Ooo0Ooo
   O0OOOo0oO . append ( [ OOooOOOO00O ] + oO )
   if 74 - 74: Ooo0Ooo * Oo + Iii1i - i1iiIII111
   if 22 - 22: IiIIii11Ii - Ooo0Ooo . i1 . i1I - ooo000
  I1IiIi = pd . DataFrame ( O0OOOo0oO , columns = [ 'Mask' , 'R' , 'G' , 'B' ] )
  if 10 - 10: Oo * I11iiIi11i1I % IiIIii11Ii / OooOoo + i1
  return I1IiIi
  if 14 - 14: I1I + I1Ii1I1
 def rgb2xyz_custom ( self , target_mask_index , reference_mask_matrix , masks , xyz_from_rgb = None ) :
  if 78 - 78: i1i1i1111I * I1Ii1I1
  oO = self . RGB_mask ( target_mask_index , reference_mask_matrix , masks )
  if 99 - 99: iI1iII1I1I1i - i1I % i1I - IiI11Ii111 * i1iiIII111 + OooOoo
  if 57 - 57: IIiIIiIi11I1
  oO = [ value / 255 for value in oO ]
  if 57 - 57: ooOOO + Ii % I11iiIi11i1I + oOo0O00 . iI1iII1I1I1i - Ii
  if 8 - 8: i1i1i1111I + i1i1i1111I * IiI11Ii111
  O00oO00oO0O = _prepare_colorarray ( oO ) . copy ( )
  if 65 - 65: iI1iII1I1I1i . Oo
  if 44 - 44: I11iiIi11i1I . I1Ii1I1 * i1 . iI1iII1I1I1i * I11iiIi11i1I - I11iiIi11i1I
  ii1 = O00oO00oO0O > 0.04045
  O00oO00oO0O [ ii1 ] = np . power ( ( O00oO00oO0O [ ii1 ] + 0.055 ) / 1.055 , 2.4 )
  O00oO00oO0O [ ~ ii1 ] /= 12.92
  if 79 - 79: IiIIii11Ii + oOo0O00
  if 50 - 50: IiI11Ii111 + i1iiIII111 . i1iiIII111 . Oo
  if xyz_from_rgb is None :
   xyz_from_rgb = np . array ( [
 [ 0.4124 , 0.3576 , 0.1805 ] ,
 [ 0.2126 , 0.7152 , 0.0722 ] ,
 [ 0.0193 , 0.1192 , 0.9505 ]
 ] )
   if 72 - 72: oOo0O00 - IiI11Ii111 + i1iiIII111 / i1I . OooOoo * IiIIii11Ii
   if 40 - 40: ooo000 * IiI11Ii111 / i1i1i1111I * oOO + i1iiIII111 - OooOoo
  IIiIIiiIIi = O00oO00oO0O @ xyz_from_rgb . T . astype ( O00oO00oO0O . dtype )
  if 29 - 29: i1i1i1111I / oOo0O00
  if 13 - 13: I11iiIi11i1I % i1iiIII111 . OooOoo % ooo000 % OooOoo
  IIiIIiiIIi *= 100
  if 21 - 21: IiI11Ii111 * I1Ii1I1
  return IIiIIiiIIi
  if 93 - 93: Ooo0Ooo . i1 + IiI11Ii111 - oOo0O00
  if 97 - 97: i1 - i1 % IIiIIiIi11I1 + IiIIii11Ii / i1I * iI1iII1I1I1i
 def generate_XYZ_dataframe ( self , reference_mask_matrix , masks , xyz_from_rgb = None ) :
  if 60 - 60: I11iiIi11i1I - Ooo0Ooo % I1Ii1I1
  ii1II1 = [ ]
  if 14 - 14: Oo % i1i1i1111I + ooOOO / I11iiIi11i1I - Ooo0Ooo / i1i1i1111I
  if 29 - 29: Iii1i + Ii - I11iiIi11i1I - IiIIii11Ii
  for OOooOOOO00O in range ( len ( masks ) ) :
   if 31 - 31: IiIIii11Ii . Iii1i / Oo + i1iiIII111
   iIoOO0oooOoo000 = self . rgb2xyz_custom ( OOooOOOO00O , reference_mask_matrix , masks , xyz_from_rgb )
   if 68 - 68: oOo0O00 . oOo0O00 / Ii . i1
   if 54 - 54: iI1iII1I1I1i % Oo . IiI11Ii111 - Iii1i % I11iiIi11i1I * i1I
   ii1II1 . append ( [ OOooOOOO00O ] + list ( iIoOO0oooOoo000 ) )
   if 31 - 31: oOo0O00 / Iii1i - IiIIii11Ii % i1I / I1Ii1I1 - i1i1i1111I
   if 68 - 68: I11iiIi11i1I . I11iiIi11i1I % I11iiIi11i1I
  I1IiIi = pd . DataFrame ( ii1II1 , columns = [ 'Mask' , 'X' , 'Y' , 'Z' ] )
  if 71 - 71: ooo000
  if 61 - 61: ooo000
  I1IiIi . to_excel ( "XYZ_masks.xlsx" , index = False )
  if 48 - 48: Iii1i * i1i1i1111I + IiIIii11Ii
  if 31 - 31: Oo * i1iiIII111 % Ii / oOO + I1Ii1I1 + iI1iII1I1I1i
  return I1IiIi
  if 90 - 90: I1Ii1I1 * i1i1i1111I / iI1iII1I1I1i * Ii
  if 38 - 38: I1I . Ii
 def rgb2lab_custom ( self , target_mask_index , reference_mask_matrix , masks , xyz_from_rgb = None ) :
  if 41 - 41: ooo000 % IIiIIiIi11I1 % ooOOO
  IIiIIiiIIi = self . rgb2xyz_custom ( target_mask_index , reference_mask_matrix , masks , xyz_from_rgb )
  if 5 - 5: oOo0O00 / Ii + i1iiIII111 * Oo + Ooo0Ooo + ooo000
  if 96 - 96: i1iiIII111 - IIiIIiIi11I1 / IIiIIiIi11I1 * IiIIii11Ii
  if 67 - 67: Ooo0Ooo . Ooo0Ooo . IiI11Ii111
  if 24 - 24: i1iiIII111 + i1i1i1111I . oOo0O00 + iI1iII1I1I1i + IiI11Ii111
  if 92 - 92: iI1iII1I1I1i / iI1iII1I1I1i + IiIIii11Ii . IiI11Ii111
  IIiIIiiIIi /= 100
  if 56 - 56: Ii * ooo000 . IiIIii11Ii
  if 66 - 66: I1Ii1I1 * OooOoo . iI1iII1I1I1i % OooOoo . i1 . IiIIii11Ii
  Oo0o0OOOOo0 = xyz2lab ( IIiIIiiIIi )
  if 18 - 18: ooo000 + ooOOO + Oo / I1Ii1I1 . IIiIIiIi11I1
  return list ( Oo0o0OOOOo0 )
  if 67 - 67: Oo % ooOOO + iI1iII1I1I1i * I1I
 def generate_LABCH_dataframe ( self , reference_mask_matrix , masks , xyz_from_rgb = None ) :
  if 79 - 79: IIiIIiIi11I1 * Oo / OooOoo
  I1Ii = [ ]
  if 88 - 88: IiIIii11Ii * i1i1i1111I * oOO
  if 70 - 70: I1Ii1I1 + I1Ii1I1 * IiIIii11Ii
  for OOooOOOO00O in range ( len ( masks ) ) :
   if 66 - 66: I1I . OooOoo
   IIOo0ooOO00O = self . rgb2lab_custom ( OOooOOOO00O , reference_mask_matrix , masks , xyz_from_rgb )
   if 68 - 68: i1I - OooOoo . i1i1i1111I + iI1iII1I1I1i
   if 60 - 60: IiIIii11Ii % oOO / i1I * OooOoo / I11iiIi11i1I - Ii
   I1Ii . append ( IIOo0ooOO00O )
   if 16 - 16: oOo0O00 / I1Ii1I1 / i1 + I11iiIi11i1I + oOo0O00
   if 11 - 11: oOO / OooOoo + oOo0O00
  I1IiIi = pd . DataFrame ( I1Ii , columns = [ 'L' , 'a' , 'b' ] )
  if 79 - 79: I11iiIi11i1I . I1Ii1I1 * i1I % I1Ii1I1 / IiI11Ii111
  if 93 - 93: i1I + Iii1i . Ii . i1I * ooOOO
  I1IiIi [ 'C' ] = sqrt ( I1IiIi [ 'a' ] ** 2 + I1IiIi [ 'b' ] ** 2 )
  I1IiIi [ 'H' ] = degrees ( arctan2 ( I1IiIi [ 'b' ] , I1IiIi [ 'a' ] ) )
  I1IiIi . loc [ I1IiIi [ 'H' ] < 0 , 'H' ] += 360
  if 84 - 84: Ooo0Ooo % IiI11Ii111
  if 82 - 82: IIiIIiIi11I1
  I1IiIi . insert ( 0 , 'Mask' , range ( len ( masks ) ) )
  if 81 - 81: oOo0O00 + i1 - ooo000 * iI1iII1I1I1i + i1i1i1111I
  if 89 - 89: I1Ii1I1
  I1IiIi . to_excel ( "LABCH_masks.xlsx" , index = False )
  if 57 - 57: iI1iII1I1I1i - i1iiIII111 / OooOoo % i1iiIII111
  if 92 - 92: IiIIii11Ii * OooOoo - IiIIii11Ii
  return I1IiIi
  if 66 - 66: i1iiIII111 . iI1iII1I1I1i / ooOOO . i1 - OooOoo
 def calculate_mask_areas ( self , masks , sort = False ) :
  if 13 - 13: oOo0O00
  OoO00 = [ ]
  if 41 - 41: IIiIIiIi11I1 / I11iiIi11i1I
  if 60 - 60: IIiIIiIi11I1 + i1i1i1111I . I11iiIi11i1I - iI1iII1I1I1i
  for OOooOOOO00O , ii1 in enumerate ( masks ) :
   if 31 - 31: IiIIii11Ii % Ii
   I111II1ii = torch . sum ( ii1 ) . item ( )
   if 56 - 56: IiIIii11Ii + ooo000
   if 47 - 47: ooo000
   OoO00 . append ( {
 'Mask' : OOooOOOO00O ,
   'Area' : I111II1ii
 } )
   if 7 - 7: I1I % oOO + i1i1i1111I + I11iiIi11i1I - iI1iII1I1I1i
   if 98 - 98: IiI11Ii111 + IiI11Ii111 * OooOoo . ooo000 . IiIIii11Ii
  Oo00 = pd . DataFrame ( OoO00 )
  if 52 - 52: Ii . OooOoo . i1iiIII111 * I11iiIi11i1I - iI1iII1I1I1i
  if 20 - 20: OooOoo % IiI11Ii111 + I1Ii1I1 + IiI11Ii111 - oOo0O00
  if sort :
   Oo00 = Oo00 . sort_values ( by = 'Area' )
   if 76 - 76: I11iiIi11i1I % IIiIIiIi11I1 % i1I
   if 39 - 39: IiIIii11Ii . Oo + i1I - oOo0O00
  Oo00 = Oo00 . reset_index ( drop = True )
  if 93 - 93: Iii1i * IIiIIiIi11I1 % i1I + i1 % Ii * i1I
  if 62 - 62: IiI11Ii111 % ooo000
  Oo00 . to_excel ( "area_masks.xlsx" , index = False )
  if 19 - 19: Ii / Oo % Iii1i / i1iiIII111 - OooOoo - Ooo0Ooo
  return Oo00
  if 89 - 89: I11iiIi11i1I - IiI11Ii111
 def plants_summary ( self , reference_mask_matrix , masks , xyz_from_rgb = None , name = None ) :
  if 61 - 61: Ii * OooOoo * I1I % i1iiIII111 % IIiIIiIi11I1 * I11iiIi11i1I
  if 49 - 49: iI1iII1I1I1i / i1iiIII111 % oOO
  if name != None :
   OOooO0OoOO0 = name
  else :
   OOooO0OoOO0 = os . path . basename ( self . image_path )
   OOooO0OoOO0 = OOooO0OoOO0 [ : - 4 ]
   if 27 - 27: IIiIIiIi11I1 * IIiIIiIi11I1 + oOO + ooOOO * Oo + i1I
   if 87 - 87: ooo000 - OooOoo + Ii + i1i1i1111I + IiIIii11Ii
  OOoo0O00Ooo00 = self . generate_RGB_dataframe ( reference_mask_matrix , masks )
  if 76 - 76: Ii / I11iiIi11i1I
  if 29 - 29: Ii / Ooo0Ooo . IIiIIiIi11I1 + OooOoo . oOO . I1Ii1I1
  O0oOOO0oo = self . generate_LABCH_dataframe ( reference_mask_matrix , masks , xyz_from_rgb )
  if 84 - 84: oOO / Ii * Ooo0Ooo / Ii / ooo000
  if 64 - 64: oOo0O00 * Ii
  I1iIII111Ii = self . calculate_mask_areas ( masks )
  if 65 - 65: I1I + i1iiIII111 . iI1iII1I1I1i / ooOOO
  if 92 - 92: i1iiIII111 . oOo0O00
  I1IiIi = pd . merge ( OOoo0O00Ooo00 , O0oOOO0oo , on = 'Mask' )
  I1IiIi = pd . merge ( I1IiIi , I1iIII111Ii , on = 'Mask' )
  if 73 - 73: I1Ii1I1 / ooo000 % I11iiIi11i1I - i1i1i1111I + Oo - I1Ii1I1
  if 18 - 18: i1 + ooOOO . i1 - iI1iII1I1I1i
  I1IiIi . insert ( 0 , 'Filename' , OOooO0OoOO0 )
  if 97 - 97: oOo0O00 + i1I % Iii1i
  if 34 - 34: i1 + Oo . oOo0O00 - ooo000 / I11iiIi11i1I * oOo0O00
  I1IiIi . to_excel ( "{0}.xlsx" . format ( OOooO0OoOO0 ) , index = False )
  if 89 - 89: oOo0O00
  if 48 - 48: i1 / IIiIIiIi11I1 / iI1iII1I1I1i / IiI11Ii111 * IiIIii11Ii
  return I1IiIi
  if 54 - 54: IIiIIiIi11I1 % I1I % IiI11Ii111 / I11iiIi11i1I . I11iiIi11i1I - IiIIii11Ii
 def pantone_summary ( self , theoretical_csv_path , mask_order , reference_mask_matrix , masks , reference_mask = None , xyz_from_rgb = None ) :
  if 10 - 10: Ii . I11iiIi11i1I % i1I / OooOoo % I1Ii1I1
  O0o = pd . read_csv ( theoretical_csv_path )
  if 92 - 92: i1i1i1111I - iI1iII1I1I1i % iI1iII1I1I1i . I11iiIi11i1I / Ooo0Ooo
  if 59 - 59: oOO . oOo0O00 - Iii1i * ooOOO - Iii1i
  OOoo0O00Ooo00 = self . generate_RGB_dataframe ( reference_mask_matrix , masks )
  iIoooOoOOoo0 = self . generate_XYZ_dataframe ( reference_mask_matrix , masks , xyz_from_rgb )
  O0oOOO0oo = self . generate_LABCH_dataframe ( reference_mask_matrix , masks , xyz_from_rgb )
  if 94 - 94: IiIIii11Ii - iI1iII1I1I1i - ooo000 - oOO . IiIIii11Ii / Oo
  if 23 - 23: I11iiIi11i1I - i1 / Iii1i . iI1iII1I1I1i + oOO
  if 55 - 55: i1 - iI1iII1I1I1i / OooOoo + I1I + Oo
  if 5 - 5: IiI11Ii111 - i1 . i1i1i1111I / IiI11Ii111 . iI1iII1I1I1i . IiI11Ii111
  if 87 - 87: i1 . Ii * iI1iII1I1I1i - oOO / Ii / OooOoo
  iIoooOoOOoo0 = iIoooOoOOoo0 . drop ( columns = [ 'Mask' ] )
  O0oOOO0oo = O0oOOO0oo . drop ( columns = [ 'Mask' ] )
  if 65 - 65: i1I / i1I + IiI11Ii111
  if 99 - 99: i1 + OooOoo + I11iiIi11i1I * Ooo0Ooo / ooOOO + Ii
  IIiI11I1I1 = pd . concat ( [ OOoo0O00Ooo00 , iIoooOoOOoo0 , O0oOOO0oo ] , axis = 1 )
  if 54 - 54: oOo0O00 / I1Ii1I1 - IIiIIiIi11I1 % i1
  if 29 - 29: OooOoo * Ii - oOO
  O0oOoOoO0o = pd . DataFrame ( { 'Mask' : mask_order , 'order' : range ( len ( mask_order ) ) } )
  if 58 - 58: oOO . I1I * Ii
  if 35 - 35: IiI11Ii111 * Ii % I1I . i1I
  O0oOoOoO0o [ 'Mask' ] = O0oOoOoO0o [ 'Mask' ] . astype ( int )
  IIiI11I1I1 [ 'Mask' ] = IIiI11I1I1 [ 'Mask' ] . astype ( int )
  if 93 - 93: I1I % I1I - oOo0O00 + I1I . Oo / IIiIIiIi11I1
  if 32 - 32: I1I % i1 - i1iiIII111 % ooo000 + oOO - IiIIii11Ii
  IIiI11I1I1 = pd . merge ( IIiI11I1I1 , O0oOoOoO0o , on = 'Mask' )
  if 40 - 40: I11iiIi11i1I + iI1iII1I1I1i - Oo
  if 93 - 93: I1I + i1iiIII111 + Ii - i1I
  IIiI11I1I1 = IIiI11I1I1 . sort_values ( 'order' ) . drop ( 'order' , axis = 1 )
  if 29 - 29: Iii1i / oOO + i1I % Ooo0Ooo * OooOoo + I1I
  if 43 - 43: i1I - i1
  OO000O = self . calculate_mask_areas ( masks )
  OO000O [ 'Mask' ] = OO000O [ 'Mask' ] . astype ( int )
  if 59 - 59: IiIIii11Ii * I1Ii1I1
  if 18 - 18: IIiIIiIi11I1 % oOO / Ii * Ii % Ii . Iii1i
  IIiI11I1I1 = pd . merge ( IIiI11I1I1 , OO000O , on = 'Mask' )
  if 53 - 53: i1 * i1i1i1111I
  if 91 - 91: I1I . I11iiIi11i1I
  O0Oo = pd . concat ( [ O0o , IIiI11I1I1 . reset_index ( drop = True ) ] , axis = 1 )
  if 69 - 69: Iii1i - I11iiIi11i1I . ooOOO * oOo0O00 % Iii1i
  if 65 - 65: Ooo0Ooo * Oo / I11iiIi11i1I . i1 / IiI11Ii111 / I11iiIi11i1I
  O0Oo [ 'ΔR' ] = np . abs ( O0Oo [ 'RT' ] - O0Oo [ 'R' ] )
  if 33 - 33: ooo000 + IiIIii11Ii . ooo000
  if 59 - 59: i1 / i1i1i1111I + Ii * ooo000 . ooo000
  O0Oo [ 'ΔG' ] = np . abs ( O0Oo [ 'GT' ] - O0Oo [ 'G' ] )
  if 49 - 49: Ii - Iii1i * I1Ii1I1 * IIiIIiIi11I1 . Ooo0Ooo
  if 83 - 83: Oo % Ii % i1
  O0Oo [ 'ΔB' ] = np . abs ( O0Oo [ 'BT' ] - O0Oo [ 'B' ] )
  if 28 - 28: IIiIIiIi11I1 % Ii + ooOOO . i1I % Ii * I1Ii1I1
  O0Oo [ 'ΔL' ] = np . abs ( O0Oo [ 'LT' ] - O0Oo [ 'L' ] )
  if 41 - 41: I1I
  O0Oo [ 'Δa' ] = np . abs ( O0Oo [ 'aT' ] - O0Oo [ 'a' ] )
  if 76 - 76: ooo000 * i1i1i1111I
  O0Oo [ 'Δb' ] = np . abs ( O0Oo [ 'bT' ] - O0Oo [ 'b' ] )
  if 39 - 39: i1 % i1I
  if 50 - 50: i1iiIII111 % OooOoo - i1i1i1111I * IiIIii11Ii % Oo . Ooo0Ooo
  O0Oo [ 'ΔE' ] = np . sqrt ( O0Oo [ 'ΔL' ] ** 2 + O0Oo [ 'Δa' ] ** 2 + O0Oo [ 'Δb' ] ** 2 )
  if 30 - 30: i1iiIII111
  if 78 - 78: ooo000 % Iii1i + ooOOO * IIiIIiIi11I1 - i1
  if reference_mask is None :
   reference_mask = mask_order [ 0 ]
   if 46 - 46: Ooo0Ooo - i1I / ooo000 * IiI11Ii111 . oOo0O00
   if 32 - 32: i1i1i1111I . OooOoo + OooOoo - ooo000 * IiIIii11Ii + Oo
  iIIIii = IIiI11I1I1 . loc [ reference_mask , [ 'L' , 'a' , 'b' ] ]
  IIiI11I1I1 [ 'distances' ] = np . sqrt ( ( IIiI11I1I1 [ 'L' ] - iIIIii [ 'L' ] ) ** 2 +
 ( IIiI11I1I1 [ 'a' ] - iIIIii [ 'a' ] ) ** 2 +
 ( IIiI11I1I1 [ 'b' ] - iIIIii [ 'b' ] ) ** 2 )
  if 12 - 12: iI1iII1I1I1i - oOO
  if 72 - 72: Oo - i1i1i1111I . oOO + IiI11Ii111 . IiI11Ii111
  O0Oo = pd . concat ( [ O0Oo , IIiI11I1I1 [ 'distances' ] . reset_index ( drop = True ) ] , axis = 1 )
  if 42 - 42: oOo0O00 . ooo000 - oOO . i1
  if 74 - 74: ooo000 / OooOoo / oOo0O00 + IIiIIiIi11I1 + OooOoo
  if 9 - 9: ooOOO - Ooo0Ooo
  OOOI1iIiiIiII1 = O0Oo [ [ 'ΔR' , 'ΔG' , 'ΔB' , 'ΔL' , 'Δa' , 'Δb' , 'ΔE' ] ] . mean ( )
  if 49 - 49: i1iiIII111 - IiI11Ii111 - OooOoo
  if 39 - 39: I1I % i1iiIII111 - I1Ii1I1
  O000O0 = pd . DataFrame ( { 'MAE' : OOOI1iIiiIiII1 . index , 'VALUES' : OOOI1iIiiIiII1 . values } )
  if 32 - 32: ooOOO - IiIIii11Ii - Oo
  if 73 - 73: oOO % I1Ii1I1
  O0Oo = O0Oo . assign ( MAE = np . nan , VALUES = np . nan )
  O0Oo . loc [ : len ( OOOI1iIiiIiII1 ) - 1 , [ 'MAE' , 'VALUES' ] ] = O000O0 . values
  if 88 - 88: iI1iII1I1I1i . I1I * oOo0O00 - i1i1i1111I * Ooo0Ooo
  if 87 - 87: oOo0O00 / ooOOO / OooOoo * Ooo0Ooo
  O0Oo . to_excel ( "pantone_summary.xlsx" , index = False )
  if 49 - 49: iI1iII1I1I1i
  return O0Oo
  if 77 - 77: ooo000 % I11iiIi11i1I % Ooo0Ooo * oOO
class I111iii11Ii11 :
 def __init__ ( self , data ) :
  if isinstance ( data , pd . DataFrame ) :
   self . data = data
  elif isinstance ( data , str ) :
   self . data = pd . read_excel ( data )
   for IIiiiIi in [ 'R' , 'G' , 'B' , 'L' , 'a' , 'b' , 'C' , 'H' , 'Area' ] :
    self . data [ IIiiiIi ] = self . data [ IIiiiIi ] . apply ( lambda III1111 : float ( str ( III1111 ) . replace ( ',' , '.' ) ) )
  else :
   raise ValueError ( 'Data should be a pandas DataFrame or a string path to an Excel file.' )
   if 20 - 20: IIiIIiIi11I1 . oOo0O00 . Ooo0Ooo
 @ property
 def PlotCIELab ( self ) :
  iiIIIIi1IIIiI = self . data . columns [ 0 ]
  if 93 - 93: i1iiIII111 - oOo0O00 . Oo . i1iiIII111 . Oo * I1Ii1I1
  if 95 - 95: I1Ii1I1 % oOO
  IIIiI1ii11 = self . data [ [ 'L' , 'a' , 'b' ] ] . values
  O0IIIi1iiI1i1 = self . data [ [ 'R' , 'G' , 'B' ] ] . values / 255.0
  if 14 - 14: IiIIii11Ii / I1I + Oo - IIiIIiIi11I1 + I11iiIi11i1I
  if 82 - 82: ooo000 % ooo000 % i1I
  II1i11I = self . data [ [ iiIIIIi1IIIiI , 'Mask' , 'L' , 'a' , 'b' , 'C' , 'H' ] ] . values
  if 79 - 79: OooOoo . Oo + oOo0O00 / I1Ii1I1 . IiIIii11Ii
  if 89 - 89: ooo000 % Ooo0Ooo
  IIIi1111iiIi1 = go . Figure ( )
  if 77 - 77: ooo000 % Ooo0Ooo
  if 24 - 24: oOO * I1Ii1I1 * I1Ii1I1 % IIiIIiIi11I1
  IIIi1111iiIi1 . add_trace ( go . Scatter3d (
 x = IIIiI1ii11 [ : , 1 ] ,
 y = IIIiI1ii11 [ : , 2 ] ,
 z = IIIiI1ii11 [ : , 0 ] ,
 customdata = II1i11I ,
  mode = 'markers' ,
 marker = dict (
 size = 6 ,
 color = [ 'rgb({},{},{})' . format ( int ( r * 255 ) , int ( g * 255 ) , int ( b * 255 ) ) for r , g , b in O0IIIi1iiI1i1 ] ,
 opacity = 1.0
 ) ,
 hovertemplate = "<b>Filename</b>: %{customdata[0]}<br><b>Mask Index</b>: %{customdata[1]}<br><b>L</b>: %{customdata[2]:.2f}<br><b>A</b>: %{customdata[3]:.2f}<br><b>B</b>: %{customdata[4]:.2f}<br><b>C</b>: %{customdata[5]:.2f}<br><b>H</b>: %{customdata[6]:.2f}<extra></extra>" ,
 ) )
  if 37 - 37: i1iiIII111 / iI1iII1I1I1i
  if 80 - 80: IiIIii11Ii
  IiI1 = np . linspace ( 0 , 2 * np . pi , 100 )
  if 65 - 65: Ooo0Ooo - IIiIIiIi11I1 * iI1iII1I1I1i - I1Ii1I1 - i1I . Iii1i
  if 98 - 98: Ii + ooOOO
  III1111 = 127 * np . cos ( IiI1 )
  II11 = 127 * np . sin ( IiI1 )
  IIIi1111iiIi1 . add_trace ( go . Scatter3d ( x = III1111 , y = II11 , z = 50 * np . ones_like ( III1111 ) , mode = 'lines' , line = dict ( color = 'black' , width = 2 , dash = 'dash' ) , showlegend = False ) )
  if 99 - 99: I11iiIi11i1I % i1I . ooOOO . oOO
  if 47 - 47: iI1iII1I1I1i . I1I % Oo - i1 * oOo0O00
  III1111 = 127 * np . cos ( IiI1 )
  IIiIii = 50 + 50 * np . sin ( IiI1 )
  IIIi1111iiIi1 . add_trace ( go . Scatter3d ( x = III1111 , y = 0 * np . ones_like ( III1111 ) , z = IIiIii , mode = 'lines' , line = dict ( color = 'black' , width = 2 , dash = 'dash' ) , showlegend = False ) )
  if 27 - 27: IiIIii11Ii % IiI11Ii111 * Ooo0Ooo * IiI11Ii111 / I1I / I1I
  if 8 - 8: oOO + IiI11Ii111 + I1Ii1I1 % oOo0O00 + oOo0O00
  II11 = 127 * np . cos ( IiI1 )
  IIiIii = 50 + 50 * np . sin ( IiI1 )
  IIIi1111iiIi1 . add_trace ( go . Scatter3d ( x = 0 * np . ones_like ( II11 ) , y = II11 , z = IIiIii , mode = 'lines' , line = dict ( color = 'black' , width = 2 , dash = 'dash' ) , showlegend = False ) )
  if 74 - 74: Oo % Ooo0Ooo / Ii
  if 36 - 36: OooOoo / OooOoo / i1iiIII111 * ooo000 / OooOoo - i1I
  OOo0oO00 = np . linspace ( 0 , 1 , 100 )
  OOO0o0o0O , oOoOoOOOOO = np . meshgrid ( np . linspace ( - 127 , 127 , 100 ) , np . linspace ( - 127 , 127 , 100 ) )
  I1iii1I = 50 * np . ones_like ( OOO0o0o0O )
  IIIi1111iiIi1 . add_trace ( go . Mesh3d ( x = OOO0o0o0O . flatten ( ) , y = oOoOoOOOOO . flatten ( ) , z = I1iii1I . flatten ( ) , opacity = 0.2 , color = 'gray' , hoverinfo = 'skip' ) )
  if 3 - 3: I1Ii1I1
  if 57 - 57: Ooo0Ooo . OooOoo / oOo0O00 * I1Ii1I1
  IIIi1111iiIi1 . add_trace ( go . Scatter3d ( x = [ 0 , 0 ] , y = [ 0 , 127 ] , z = [ 50 , 50 ] , mode = 'lines' , line = dict ( color = 'black' , width = 2 ) , showlegend = False ) )
  IIIi1111iiIi1 . add_trace ( go . Scatter3d ( x = [ 0 , 0 ] , y = [ 0 , - 127 ] , z = [ 50 , 50 ] , mode = 'lines' , line = dict ( color = 'black' , width = 2 ) , showlegend = False ) )
  IIIi1111iiIi1 . add_trace ( go . Scatter3d ( x = [ 0 , 0 ] , y = [ 0 , 0 ] , z = [ 100 , 50 ] , mode = 'lines' , line = dict ( color = 'black' , width = 2 ) , showlegend = False ) )
  IIIi1111iiIi1 . add_trace ( go . Scatter3d ( x = [ 0 , 0 ] , y = [ 0 , 0 ] , z = [ 0 , 50 ] , mode = 'lines' , line = dict ( color = 'black' , width = 2 ) , showlegend = False ) )
  IIIi1111iiIi1 . add_trace ( go . Scatter3d ( x = [ 0 , 127 ] , y = [ 0 , 0 ] , z = [ 50 , 50 ] , mode = 'lines' , line = dict ( color = 'black' , width = 2 ) , showlegend = False ) )
  IIIi1111iiIi1 . add_trace ( go . Scatter3d ( x = [ 0 , - 127 ] , y = [ 0 , 0 ] , z = [ 50 , 50 ] , mode = 'lines' , line = dict ( color = 'black' , width = 2 ) , showlegend = False ) )
  if 36 - 36: IiIIii11Ii
  if 33 - 33: IiIIii11Ii
  IIIi1111iiIi1 . update_layout (
 autosize = False ,
 width = 1000 ,
 height = 1000 ,
 scene = dict (
 xaxis = dict ( range = [ - 128 , 128 ] , title = 'a' ) ,
 yaxis = dict ( range = [ - 128 , 128 ] , title = 'b' ) ,
 zaxis = dict ( range = [ 0 , 100 ] , title = 'L' )
 ) ,
 )
  if 86 - 86: Ii / oOo0O00 - i1 * i1i1i1111I - Oo * Iii1i
  if 28 - 28: OooOoo . i1 % iI1iII1I1I1i % Iii1i
  IIIi1111iiIi1 . show ( )
  if 2 - 2: Oo + OooOoo - i1i1i1111I - ooo000 / Oo . Oo
  if 41 - 41: OooOoo + OooOoo - OooOoo
 @ property
 def Plotab ( self ) :
  iiIIIIi1IIIiI = self . data . columns [ 0 ]
  if 9 - 9: Iii1i % I1Ii1I1 % IiIIii11Ii - I1I * OooOoo
  if 53 - 53: iI1iII1I1I1i * i1i1i1111I / OooOoo . i1I . ooo000
  OooO0oo = self . data [ [ 'a' , 'b' ] ] . values
  O0IIIi1iiI1i1 = self . data [ [ 'R' , 'G' , 'B' ] ] . values / 255.0
  if 71 - 71: IiI11Ii111 % Iii1i + oOo0O00 * I11iiIi11i1I / IiI11Ii111
  if 66 - 66: ooo000 * oOo0O00
  II1i11I = self . data [ [ iiIIIIi1IIIiI , 'Mask' , 'L' , 'a' , 'b' , 'C' , 'H' ] ] . values
  if 83 - 83: i1
  if 64 - 64: i1 . IiI11Ii111 - I11iiIi11i1I . Iii1i
  IIIi1111iiIi1 = go . Figure ( )
  if 47 - 47: ooo000 / Ooo0Ooo % IiIIii11Ii
  if 70 - 70: IIiIIiIi11I1 / Iii1i . i1i1i1111I % ooOOO . Ii / Ii
  IIIi1111iiIi1 . add_trace ( go . Scatter (
 x = OooO0oo [ : , 0 ] ,
 y = OooO0oo [ : , 1 ] ,
 customdata = II1i11I ,
  mode = 'markers' ,
 marker = dict (
 size = 6 ,
 color = [ 'rgb({},{},{})' . format ( int ( r * 255 ) , int ( g * 255 ) , int ( b * 255 ) ) for r , g , b in O0IIIi1iiI1i1 ] ,
 opacity = 1.0
 ) ,
 hovertemplate = "<b>Filename</b>: %{customdata[0]}<br><b>Mask Index</b>: %{customdata[1]}<br><b>L</b>: %{customdata[2]:.2f}<br><b>A</b>: %{customdata[3]:.2f}<br><b>B</b>: %{customdata[4]:.2f}<br><b>C</b>: %{customdata[5]:.2f}<br><b>H</b>: %{customdata[6]:.2f}<extra></extra>" ,
 ) )
  if 51 - 51: I11iiIi11i1I + Ooo0Ooo - ooo000 * oOO . oOO
  if 79 - 79: i1i1i1111I + oOo0O00
  IIIi1111iiIi1 . add_trace ( go . Scatter ( x = [ 0 , 0 ] , y = [ - 127 , 127 ] , mode = 'lines' , line = dict ( color = 'black' , width = 1 ) ) )
  IIIi1111iiIi1 . add_trace ( go . Scatter ( x = [ - 127 , 127 ] , y = [ 0 , 0 ] , mode = 'lines' , line = dict ( color = 'black' , width = 1 ) ) )
  if 11 - 11: OooOoo / i1iiIII111 % i1I - i1i1i1111I * oOo0O00
  if 90 - 90: i1 * i1 . Ooo0Ooo . Oo
  for IiI1 in np . arange ( 0 , 2 * np . pi , np . pi / 18 ) :
   o0000o = 127 * np . cos ( IiI1 )
   ii1iiII111IiI = 127 * np . sin ( IiI1 )
   IIIi1111iiIi1 . add_trace ( go . Scatter ( x = [ 0 , o0000o ] , y = [ 0 , ii1iiII111IiI ] , mode = 'lines' ,
 line = dict ( color = 'rgba(128, 128, 128, 0.5)' , width = 1 , dash = 'dot' ) , showlegend = False ) )
   if 17 - 17: Oo + i1iiIII111 . ooOOO - i1i1i1111I % i1I + i1i1i1111I
   if 64 - 64: IiIIii11Ii % IIiIIiIi11I1 . I1Ii1I1 % IiIIii11Ii
  IIIi1111iiIi1 . add_shape (
 type = "circle" ,
 xref = "x" , yref = "y" ,
 x0 = - 127 , y0 = - 127 , x1 = 127 , y1 = 127 ,
 line = dict ( color = "black" , width = 2 , dash = "dash" ) ,
 )
  if 21 - 21: i1I . Oo * i1
  if 95 - 95: Ii . i1 . IIiIIiIi11I1 . ooo000 + OooOoo
  IIIi1111iiIi1 . update_layout (
 xaxis = dict ( range = [ - 128 , 128 ] , title = 'a' ) ,
 yaxis = dict ( range = [ - 128 , 128 ] , title = 'b' ) ,
 autosize = False ,
 width = 800 ,
 height = 800 ,
 )
  if 23 - 23: oOO
  if 92 - 92: Ooo0Ooo * i1iiIII111 / Ooo0Ooo / i1iiIII111 * i1i1i1111I + i1I
  IIIi1111iiIi1 . show ( )
  if 48 - 48: I11iiIi11i1I
 @ property
 def PlotaL ( self ) :
  iiIIIIi1IIIiI = self . data . columns [ 0 ]
  if 87 - 87: I1Ii1I1
  iiIIIIi1 = self . data [ [ 'a' , 'L' ] ] . values
  O0IIIi1iiI1i1 = self . data [ [ 'R' , 'G' , 'B' ] ] . values / 255.0
  II1i11I = self . data [ [ iiIIIIi1IIIiI , 'Mask' , 'L' , 'a' , 'b' , 'C' , 'H' ] ] . values
  if 59 - 59: IiIIii11Ii
  IIIi1111iiIi1 = go . Figure ( )
  IIIi1111iiIi1 . add_trace ( go . Scatter (
 x = iiIIIIi1 [ : , 0 ] ,
 y = iiIIIIi1 [ : , 1 ] ,
 customdata = II1i11I ,
 mode = 'markers' ,
 marker = dict (
 size = 6 ,
 color = [ 'rgb({},{},{})' . format ( int ( r * 255 ) , int ( g * 255 ) , int ( II11iII1Ii1 * 255 ) ) for r , g , II11iII1Ii1 in O0IIIi1iiI1i1 ] ,
 opacity = 1.0
 ) ,
 hovertemplate = "<b>Filename</b>: %{customdata[0]}<br><b>Mask Index</b>: %{customdata[1]}<br><b>L</b>: %{customdata[2]:.2f}<br><b>A</b>: %{customdata[3]:.2f}<br><b>B</b>: %{customdata[4]:.2f}<br><b>C</b>: %{customdata[5]:.2f}<br><b>H</b>: %{customdata[6]:.2f}<extra></extra>" ,
 ) )
  if 70 - 70: Iii1i / Ooo0Ooo
  IIIi1111iiIi1 . add_trace ( go . Scatter ( x = [ 0 , 0 ] , y = [ 0 , 100 ] , mode = 'lines' , line = dict ( color = 'black' , width = 1 ) ) )
  IIIi1111iiIi1 . add_trace ( go . Scatter ( x = [ - 127 , 127 ] , y = [ 50 , 50 ] , mode = 'lines' , line = dict ( color = 'black' , width = 1 ) ) )
  if 4 - 4: IiIIii11Ii % Iii1i
  if 43 - 43: ooo000
  IiI1 = np . linspace ( 0 , 2 * np . pi , 100 )
  o0oOO = 127
  II11iII1Ii1 = 50
  III1111 = o0oOO * np . cos ( IiI1 )
  II11 = II11iII1Ii1 * np . sin ( IiI1 ) + 50
  IIIi1111iiIi1 . add_trace ( go . Scatter ( x = III1111 , y = II11 , mode = 'lines' , line = dict ( color = 'black' , width = 2 , dash = 'dash' ) ) )
  if 8 - 8: ooOOO * i1i1i1111I
  IIIi1111iiIi1 . update_layout (
 xaxis = dict ( range = [ - 128 , 128 ] , title = 'a' ) ,
 yaxis = dict ( range = [ 0 , 100 ] , title = 'L' ) ,
 autosize = False ,
 width = 800 ,
 height = 800 ,
 )
  IIIi1111iiIi1 . show ( )
  if 6 - 6: Ooo0Ooo % IiIIii11Ii
 @ property
 def elbow_method ( self ) :
  if 59 - 59: i1I . I11iiIi11i1I . ooo000 / i1I . Ii % Ooo0Ooo
  iIII11 = self . data [ [ 'L' , 'a' , 'b' ] ] . values
  if 42 - 42: ooOOO - i1i1i1111I / I1I
  if 18 - 18: OooOoo . I1Ii1I1 . iI1iII1I1I1i - Ooo0Ooo + Oo
  oOOOooo0O = SimpleImputer ( missing_values = np . nan , strategy = 'mean' )
  iii1iII = oOOOooo0O . fit_transform ( iIII11 )
  if 66 - 66: oOo0O00
  if 34 - 34: ooo000 . iI1iII1I1I1i
  iiiiiiiii = [ ]
  O00o00Ooooo0 = range ( 1 , 11 )
  for oOo in O00o00Ooooo0 :
   IiOoO0OoOoo = KMeans ( n_clusters = oOo )
   IiOoO0OoOoo . fit ( iii1iII )
   iiiiiiiii . append ( sum ( np . min ( cdist ( iii1iII , IiOoO0OoOoo . cluster_centers_ , 'euclidean' ) , axis = 1 ) ) / iii1iII . shape [ 0 ] )
   if 36 - 36: OooOoo - IiI11Ii111 . IiI11Ii111
   if 79 - 79: oOo0O00 * IIiIIiIi11I1
  OoOOo0 = [ 0 ] + [ iiiiiiiii [ i ] - 2 * iiiiiiiii [ i - 1 ] + iiiiiiiii [ i - 2 ] for i in range ( 2 , len ( iiiiiiiii ) ) ]
  if 68 - 68: oOO * Iii1i
  if 49 - 49: IIiIIiIi11I1 + IIiIIiIi11I1 % i1I + i1iiIII111 % ooOOO - I1Ii1I1
  IiiIiI1I1ii = np . argmax ( OoOOo0 ) + 1
  if 13 - 13: i1 * ooOOO
  print ( 'Optimal number of clusters: {0}' . format ( IiiIiI1I1ii ) )
  if 69 - 69: i1 . ooo000 % I1Ii1I1 + i1 / i1iiIII111 % IiIIii11Ii
  if 76 - 76: IiIIii11Ii
  IIIi1111iiIi1 = go . Figure ( data = go . Scatter ( x = list ( O00o00Ooooo0 ) , y = iiiiiiiii , mode = 'lines+markers' ) )
  if 73 - 73: i1I / IIiIIiIi11I1 / oOo0O00 % IIiIIiIi11I1 / Ii
  if 12 - 12: i1
  IIIi1111iiIi1 . update_layout ( title = 'Elbow Method' ,
 xaxis = dict ( title = 'Number of clusters (k)' ) ,
 yaxis = dict ( title = 'Distortion Score' ) )
  if 78 - 78: I11iiIi11i1I . IIiIIiIi11I1
  IIIi1111iiIi1 . show ( )
  if 1 - 1: I1I % i1iiIII111 / I1Ii1I1
 def clusters ( self , n_clusters ) :
  if 42 - 42: Iii1i
  iIII11 = self . data [ [ 'L' , 'a' , 'b' ] ] . values
  if 19 - 19: OooOoo * I11iiIi11i1I . i1i1i1111I + ooo000 + i1I
  if 13 - 13: i1iiIII111 + I1Ii1I1 / Ooo0Ooo
  oOOOooo0O = SimpleImputer ( missing_values = np . nan , strategy = 'mean' )
  iii1iII = oOOOooo0O . fit_transform ( iIII11 )
  if 82 - 82: i1iiIII111 * oOO / IiI11Ii111 * IiIIii11Ii
  if 29 - 29: I1Ii1I1 % oOo0O00 - oOo0O00 * I1Ii1I1 / Iii1i * i1I
  o0O0 = KMeans ( n_clusters = n_clusters , random_state = 0 ) . fit ( iii1iII )
  if 29 - 29: Oo % OooOoo - i1i1i1111I + IiIIii11Ii / I1Ii1I1 + i1i1i1111I
  if 91 - 91: i1iiIII111 * Iii1i - I1I + i1i1i1111I / ooOOO + Oo
  iI1IiIII = self . data . copy ( )
  if 45 - 45: Oo - i1I - i1 * iI1iII1I1I1i / i1i1i1111I
  if 5 - 5: i1iiIII111 . Oo + i1i1i1111I
  iI1IiIII [ 'cluster' ] = o0O0 . labels_
  if 44 - 44: I1Ii1I1 - i1i1i1111I
  if 71 - 71: i1iiIII111
  iI1IiIII . to_excel ( 'cluster_data.xlsx' , index = False )
  if 77 - 77: Iii1i - I1Ii1I1 - Ooo0Ooo % Ii / Oo
  if 43 - 43: Ooo0Ooo / oOO
  return iI1IiIII
  if 93 - 93: ooOOO % oOo0O00 * i1 + Ii . i1I - oOO
 def PlotClusterCIELab ( self , n_clusters , distinction = None ) :
  if 63 - 63: IiIIii11Ii . Iii1i
  o000Oo = self . clusters ( n_clusters )
  if 13 - 13: I1I
  if 22 - 22: i1i1i1111I + Oo - ooo000 + IIiIIiIi11I1 - ooOOO * i1i1i1111I
  IIIiI1ii11 = o000Oo [ [ 'L' , 'a' , 'b' ] ] . values
  if 83 - 83: I1Ii1I1 - ooo000 + I1Ii1I1 + Ii - ooOOO
  if 57 - 57: i1 + i1I + OooOoo . Oo
  II1i11I = o000Oo [ [ 'Filename' , 'Mask' , 'L' , 'a' , 'b' , 'C' , 'H' , 'cluster' ] ] . values
  if 11 - 11: oOO + i1I - oOO / ooo000 . Ooo0Ooo % oOO
  if 88 - 88: i1 % oOo0O00 - ooo000 . Ii
  IIIi1111iiIi1 = go . Figure ( )
  if 65 - 65: i1I % IiI11Ii111 * Oo / I1Ii1I1 - I1Ii1I1
  if 28 - 28: Ii / Ooo0Ooo
  I1i1I = [ 'circle' , 'cross' , 'square' , 'x' , 'diamond' ]
  if 41 - 41: i1i1i1111I . oOO - oOO - ooOOO + iI1iII1I1I1i % Iii1i
  if 90 - 90: I11iiIi11i1I % Oo - OooOoo - i1I / i1I
  iiI1IIII1I1 = [ ]
  if 78 - 78: I1Ii1I1 . ooo000 + IiI11Ii111 / IiI11Ii111
  if 88 - 88: Ii - i1iiIII111 - i1 * IIiIIiIi11I1 . IiIIii11Ii + Iii1i
  for OOooOOOO00O in range ( n_clusters ) :
   ii1 = II1i11I [ : , 7 ] == OOooOOOO00O
   OOO00o = [ ]
   for OOooO0OoOO0 , i1III1 , oOoO0 , II11iII1Ii1 in zip ( II1i11I [ ii1 , 0 ] , o000Oo [ 'R' ] [ ii1 ] , o000Oo [ 'G' ] [ ii1 ] , o000Oo [ 'B' ] [ ii1 ] ) :
    if distinction is not None and OOooO0OoOO0 in distinction :
     OOO00o . append ( distinction [ OOooO0OoOO0 ] )
    else :
     OOO00o . append ( 'rgb({},{},{})' . format ( int ( i1III1 ) , int ( oOoO0 ) , int ( II11iII1Ii1 ) ) )
     if 63 - 63: Ii + I1Ii1I1 % oOo0O00 . I11iiIi11i1I * ooo000
   iiI1IIII1I1 . append ( go . Scatter3d (
 x = IIIiI1ii11 [ ii1 , 1 ] ,
 y = IIIiI1ii11 [ ii1 , 2 ] ,
 z = IIIiI1ii11 [ ii1 , 0 ] ,
 customdata = II1i11I [ ii1 ] ,
   mode = 'markers' ,
 marker = dict (
 size = 6 ,
 color = OOO00o ,
 opacity = 1.0 ,
 symbol = I1i1I [ OOooOOOO00O % len ( I1i1I ) ]
   ) ,
 hovertemplate = "<b>Filename</b>: %{customdata[0]}<br><b>Mask Index</b>: %{customdata[1]}<br><b>L</b>: %{customdata[2]:.2f}<br><b>A</b>: %{customdata[3]:.2f}<br><b>B</b>: %{customdata[4]:.2f}<br><b>C</b>: %{customdata[5]:.2f}<br><b>H</b>: %{customdata[6]:.2f}<br><b>Cluster</b>: %{customdata[7]}<extra></extra>" ,
 ) )
   if 18 - 18: Iii1i % I1I / Ii / Ii / iI1iII1I1I1i
   if 7 - 7: I1I * IiIIii11Ii + iI1iII1I1I1i
  IiI1 = np . linspace ( 0 , 2 * np . pi , 100 )
  if 65 - 65: Oo * Iii1i * IiIIii11Ii / IIiIIiIi11I1
  if 77 - 77: IIiIIiIi11I1 / oOO % IiIIii11Ii + I1Ii1I1
  III1111 = 127 * np . cos ( IiI1 )
  II11 = 127 * np . sin ( IiI1 )
  IIIi1111iiIi1 . add_trace ( go . Scatter3d ( x = III1111 , y = II11 , z = 50 * np . ones_like ( III1111 ) , mode = 'lines' , line = dict ( color = 'black' , width = 2 , dash = 'dash' ) , showlegend = False ) )
  if 34 - 34: i1i1i1111I * Iii1i + Ii - i1I / i1 / i1
  if 87 - 87: Ooo0Ooo + I11iiIi11i1I
  III1111 = 127 * np . cos ( IiI1 )
  IIiIii = 50 + 50 * np . sin ( IiI1 )
  IIIi1111iiIi1 . add_trace ( go . Scatter3d ( x = III1111 , y = 0 * np . ones_like ( III1111 ) , z = IIiIii , mode = 'lines' , line = dict ( color = 'black' , width = 2 , dash = 'dash' ) , showlegend = False ) )
  if 3 - 3: IIiIIiIi11I1 . I1Ii1I1 / Iii1i % ooOOO / oOo0O00 % ooOOO
  if 55 - 55: iI1iII1I1I1i + I1I + oOo0O00 - I11iiIi11i1I
  II11 = 127 * np . cos ( IiI1 )
  IIiIii = 50 + 50 * np . sin ( IiI1 )
  IIIi1111iiIi1 . add_trace ( go . Scatter3d ( x = 0 * np . ones_like ( II11 ) , y = II11 , z = IIiIii , mode = 'lines' , line = dict ( color = 'black' , width = 2 , dash = 'dash' ) , showlegend = False ) )
  if 80 - 80: Oo % I11iiIi11i1I . IiIIii11Ii * oOO
  if 83 - 83: oOo0O00 % Oo + i1i1i1111I - ooOOO + i1iiIII111
  OOo0oO00 = np . linspace ( 0 , 1 , 100 )
  OOO0o0o0O , oOoOoOOOOO = np . meshgrid ( np . linspace ( - 127 , 127 , 100 ) , np . linspace ( - 127 , 127 , 100 ) )
  I1iii1I = 50 * np . ones_like ( OOO0o0o0O )
  IIIi1111iiIi1 . add_trace ( go . Mesh3d ( x = OOO0o0o0O . flatten ( ) , y = oOoOoOOOOO . flatten ( ) , z = I1iii1I . flatten ( ) , opacity = 0.2 , color = 'gray' , hoverinfo = 'skip' ) )
  if 36 - 36: oOo0O00 * OooOoo / I11iiIi11i1I
  if 98 - 98: IiI11Ii111 . ooo000
  for oo0OOOOOO0o00 in iiI1IIII1I1 :
   IIIi1111iiIi1 . add_trace ( oo0OOOOOO0o00 )
   if 74 - 74: i1 + oOO
   if 82 - 82: IiIIii11Ii * oOo0O00 % oOO / I1Ii1I1
  IIIi1111iiIi1 . update_layout (
 autosize = False ,
 width = 1000 ,
 height = 1000 ,
 scene = dict (
 xaxis = dict ( range = [ - 128 , 128 ] , title = 'a' ) ,
 yaxis = dict ( range = [ - 128 , 128 ] , title = 'b' ) ,
 zaxis = dict ( range = [ 0 , 100 ] , title = 'L' )
 ) ,
 )
  if 38 - 38: IIiIIiIi11I1
  if 17 - 17: oOO + i1iiIII111 / i1i1i1111I * IiIIii11Ii / oOo0O00 / Ooo0Ooo
  IIIi1111iiIi1 . show ( )
  if 43 - 43: i1I + iI1iII1I1I1i + Oo
 def PlotClusterab ( self , n_clusters , distinction = None ) :
  if 61 - 61: i1I
  o000Oo = self . clusters ( n_clusters )
  if 75 - 75: IIiIIiIi11I1 . I11iiIi11i1I / ooo000 . oOo0O00 % Ooo0Ooo
  if 7 - 7: I1I / i1iiIII111
  OooO0oo = o000Oo [ [ 'a' , 'b' ] ] . values
  if 53 - 53: OooOoo - I1Ii1I1 + IIiIIiIi11I1 . I1Ii1I1 + i1iiIII111
  if 82 - 82: Iii1i
  II1i11I = o000Oo [ [ 'Filename' , 'Mask' , 'L' , 'a' , 'b' , 'C' , 'H' , 'cluster' ] ] . values
  if 58 - 58: I1Ii1I1 / I1Ii1I1 % IIiIIiIi11I1
  if 98 - 98: OooOoo
  IIIi1111iiIi1 = go . Figure ( )
  if 79 - 79: Ooo0Ooo * Ooo0Ooo - ooo000 - Iii1i
  if 5 - 5: IiIIii11Ii / i1iiIII111 . I1I % Iii1i . i1
  I1i1I = [ 'circle' , 'cross' , 'square' , 'x' , 'diamond' ]
  if 29 - 29: oOo0O00 + Oo / I11iiIi11i1I - oOo0O00 % i1i1i1111I / Ii
  if 39 - 39: I11iiIi11i1I - I11iiIi11i1I % i1iiIII111
  for OOooOOOO00O in range ( n_clusters ) :
   ii1 = II1i11I [ : , 7 ] == OOooOOOO00O
   OOO00o = [ ]
   for OOooO0OoOO0 , i1III1 , oOoO0 , II11iII1Ii1 in zip ( II1i11I [ ii1 , 0 ] , o000Oo [ 'R' ] [ ii1 ] , o000Oo [ 'G' ] [ ii1 ] , o000Oo [ 'B' ] [ ii1 ] ) :
    if distinction is not None and OOooO0OoOO0 in distinction :
     OOO00o . append ( distinction [ OOooO0OoOO0 ] )
    else :
     OOO00o . append ( 'rgb({},{},{})' . format ( int ( i1III1 ) , int ( oOoO0 ) , int ( II11iII1Ii1 ) ) )
     if 56 - 56: i1iiIII111 / IiIIii11Ii - I1Ii1I1 / Iii1i - i1 / OooOoo
   IIIi1111iiIi1 . add_trace ( go . Scatter (
 x = OooO0oo [ ii1 , 0 ] ,
 y = OooO0oo [ ii1 , 1 ] ,
 customdata = II1i11I [ ii1 ] ,
   mode = 'markers' ,
 marker = dict (
 size = 6 ,
 color = OOO00o ,
 opacity = 1.0 ,
 symbol = I1i1I [ OOooOOOO00O % len ( I1i1I ) ]
   ) ,
 hovertemplate = "<b>Filename</b>: %{customdata[0]}<br><b>Mask Index</b>: %{customdata[1]}<br><b>L</b>: %{customdata[2]:.2f}<br><b>A</b>: %{customdata[3]:.2f}<br><b>B</b>: %{customdata[4]:.2f}<br><b>C</b>: %{customdata[5]:.2f}<br><b>H</b>: %{customdata[6]:.2f}<br><b>Cluster</b>: %{customdata[7]}<extra></extra>" ,
 ) )
   if 28 - 28: i1I . I1Ii1I1 % IiIIii11Ii . Ii * ooOOO - oOO
   if 80 - 80: IiI11Ii111
  IIIi1111iiIi1 . add_trace ( go . Scatter ( x = [ 0 , 0 ] , y = [ - 127 , 127 ] , mode = 'lines' , line = dict ( color = 'black' , width = 1 ) ) )
  IIIi1111iiIi1 . add_trace ( go . Scatter ( x = [ - 127 , 127 ] , y = [ 0 , 0 ] , mode = 'lines' , line = dict ( color = 'black' , width = 1 ) ) )
  if 7 - 7: Iii1i . Oo . iI1iII1I1I1i - Oo - ooOOO % IIiIIiIi11I1
  if 90 - 90: Iii1i / oOo0O00
  for IiI1 in np . arange ( 0 , 2 * np . pi , np . pi / 18 ) :
   o0000o = 127 * np . cos ( IiI1 )
   ii1iiII111IiI = 127 * np . sin ( IiI1 )
   IIIi1111iiIi1 . add_trace ( go . Scatter ( x = [ 0 , o0000o ] , y = [ 0 , ii1iiII111IiI ] , mode = 'lines' ,
 line = dict ( color = 'rgba(128, 128, 128, 0.5)' , width = 1 , dash = 'dot' ) , showlegend = False ) )
   if 54 - 54: I11iiIi11i1I . IiI11Ii111 + OooOoo % Oo - IiI11Ii111 % Oo
   if 37 - 37: Oo
  IIIi1111iiIi1 . add_shape (
 type = "circle" ,
 xref = "x" , yref = "y" ,
 x0 = - 127 , y0 = - 127 , x1 = 127 , y1 = 127 ,
 line = dict ( color = "black" , width = 2 , dash = "dash" ) ,
 )
  if 64 - 64: I1I % ooOOO
  if 67 - 67: Ooo0Ooo + oOo0O00
  IIIi1111iiIi1 . update_layout (
 xaxis = dict ( range = [ - 128 , 128 ] , title = 'a' ) ,
 yaxis = dict ( range = [ - 128 , 128 ] , title = 'b' ) ,
 autosize = False ,
 width = 800 ,
 height = 800 ,
 )
  if 21 - 21: oOO
  if 64 - 64: IIiIIiIi11I1 * i1iiIII111
  IIIi1111iiIi1 . show ( )
  if 31 - 31: Iii1i
  if 86 - 86: Oo
 def PlotClusteraL ( data , n_clusters , distinction = None ) :
  if 74 - 74: OooOoo - I11iiIi11i1I . Oo . i1 + oOo0O00 - ooOOO
  o000Oo = data . clusters ( n_clusters )
  if 60 - 60: Oo
  if 23 - 23: oOo0O00
  O0ooooo0 = o000Oo [ [ 'a' , 'L' ] ] . values
  if 83 - 83: IiIIii11Ii + iI1iII1I1I1i
  if 80 - 80: i1i1i1111I * i1iiIII111 + I1Ii1I1
  II1i11I = o000Oo [ [ 'Filename' , 'Mask' , 'L' , 'a' , 'b' , 'C' , 'H' , 'cluster' ] ] . values
  if 15 - 15: iI1iII1I1I1i - ooo000 + IIiIIiIi11I1 * Ooo0Ooo
  if 84 - 84: OooOoo . Ii
  IIIi1111iiIi1 = go . Figure ( )
  if 21 - 21: Ooo0Ooo
  if 76 - 76: iI1iII1I1I1i % IiI11Ii111 / IiIIii11Ii - IiI11Ii111
  I1i1I = [ 'circle' , 'cross' , 'square' , 'x' , 'diamond' ]
  if 4 - 4: OooOoo * IIiIIiIi11I1 - Ooo0Ooo . Ii - ooOOO . i1iiIII111
  if 2 - 2: I11iiIi11i1I * i1I * i1I * i1iiIII111 / I11iiIi11i1I % IiI11Ii111
  for OOooOOOO00O in range ( n_clusters ) :
   ii1 = II1i11I [ : , 7 ] == OOooOOOO00O
   OOO00o = [ ]
   for OOooO0OoOO0 , i1III1 , oOoO0 , II11iII1Ii1 in zip ( II1i11I [ ii1 , 0 ] , o000Oo [ 'R' ] [ ii1 ] , o000Oo [ 'G' ] [ ii1 ] , o000Oo [ 'B' ] [ ii1 ] ) :
    if distinction is not None and OOooO0OoOO0 in distinction :
     OOO00o . append ( distinction [ OOooO0OoOO0 ] )
    else :
     OOO00o . append ( 'rgb({},{},{})' . format ( int ( i1III1 ) , int ( oOoO0 ) , int ( II11iII1Ii1 ) ) )
     if 13 - 13: i1 * IiIIii11Ii - ooo000 * OooOoo
   IIIi1111iiIi1 . add_trace ( go . Scatter (
 x = O0ooooo0 [ ii1 , 0 ] ,
 y = O0ooooo0 [ ii1 , 1 ] ,
 customdata = II1i11I [ ii1 ] ,
   mode = 'markers' ,
 marker = dict (
 size = 6 ,
 color = OOO00o ,
 opacity = 1.0 ,
 symbol = I1i1I [ OOooOOOO00O % len ( I1i1I ) ]
   ) ,
 hovertemplate = "<b>Filename</b>: %{customdata[0]}<br><b>Mask Index</b>: %{customdata[1]}<br><b>L</b>: %{customdata[2]:.2f}<br><b>A</b>: %{customdata[3]:.2f}<br><b>B</b>: %{customdata[4]:.2f}<br><b>C</b>: %{customdata[5]:.2f}<br><b>H</b>: %{customdata[6]:.2f}<br><b>Cluster</b>: %{customdata[7]}<extra></extra>" ,
 ) )
   if 72 - 72: Iii1i . I1Ii1I1 * IIiIIiIi11I1 % I1Ii1I1 - i1i1i1111I
   if 97 - 97: i1 + ooOOO + I11iiIi11i1I . i1
  IIIi1111iiIi1 . add_trace ( go . Scatter ( x = [ 0 , 0 ] , y = [ 0 , 100 ] , mode = 'lines' , line = dict ( color = 'black' , width = 1 ) ) )
  IIIi1111iiIi1 . add_trace ( go . Scatter ( x = [ - 127 , 127 ] , y = [ 50 , 50 ] , mode = 'lines' , line = dict ( color = 'black' , width = 1 ) ) )
  if 71 - 71: ooo000 - i1i1i1111I / iI1iII1I1I1i - iI1iII1I1I1i % IIiIIiIi11I1 . iI1iII1I1I1i
  if 41 - 41: IIiIIiIi11I1
  IiI1 = np . linspace ( 0 , 2 * np . pi , 100 )
  o0oOO = 127
  II11iII1Ii1 = 50
  III1111 = o0oOO * np . cos ( IiI1 )
  II11 = II11iII1Ii1 * np . sin ( IiI1 ) + 50
  IIIi1111iiIi1 . add_trace ( go . Scatter ( x = III1111 , y = II11 , mode = 'lines' , line = dict ( color = 'black' , width = 2 , dash = 'dash' ) ) )
  if 96 - 96: Oo . Ooo0Ooo - oOO . Iii1i % IiI11Ii111 . IiI11Ii111
  if 82 - 82: oOO % iI1iII1I1I1i / i1iiIII111
  IIIi1111iiIi1 . update_layout (
 xaxis = dict ( range = [ - 128 , 128 ] , title = 'a' ) ,
 yaxis = dict ( range = [ 0 , 100 ] , title = 'L' ) ,
 autosize = False ,
 width = 800 ,
 height = 800 ,
 )
  if 47 - 47: i1
  if 59 - 59: IiI11Ii111 % IIiIIiIi11I1 % IiIIii11Ii
  IIIi1111iiIi1 . show ( )
  if 92 - 92: Oo . i1 * i1iiIII111 . IiIIii11Ii
# dd678faae9ac167bc83abf78e5cb2f3f0688d3a3
