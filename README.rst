PyColorimetry
=============

**PyColorimetry** is a highly versatile Python library designed to perform colorimetry analysis on images. Leveraging OpenAI's LangSAM model, this library allows you to extract and analyze color information based on textual prompts.

This library is perfect for researchers, digital artists, designers, and anyone else who needs precise and detailed color information from images.

Key Features
------------

1. Extract color information from images based on textual prompts.
2. Generate color masks for specific color areas in an image.

Installation
============

You can install PyColorimetry using pip...

.. code:: bash

    pip install PyColorimetry

Example Usage
=============

Here is a simple example of how to use PyColorimetry:

.. code:: python

    from PyColorimetry import Images

    image = Images('path_to_your_image.jpg')
    image.show

    text_prompt = "small white rectangle"
    reference_mask_image  = image.reference_mask(text_prompt, mask_index=0, matrix = False)
    reference_mask_matrix = image.reference_mask(text_prompt, mask_index=0, matrix = True)

    df = image.summary(reference_mask_matrix, masks)
    print(df)
