## generate.py 

`generate.py` acts as the command line parser and containts default training parameters.

**Flags.**

- `base_img_path`: path to base image (required).
- `style_img_path`: path to style image (required).
- `output_img_path`: path location for saving output (required).
- `--iters`: number of iterations for L-BFGS optimizer. Default has been set to `100`.
- `--content_weight`: weight for content feature loss. Default has been set to `7.5e0`.
- `--style_weight`:  weight for style feature loss. Default has been set to `1e2`.
- `--tv_weight`: weight for total variation loss. Default has been set to `2e2`.
- `--width`: output image width. If specified, output image height is changed accordingly.
- `--convnet`: vgg model to use: 16 or 19. Default has been set to `vgg16`.


## neural_styler.py

`Neural_Styler()` is a class abstraction that implements the logic of the algorithm. It calculates the 3 loss functions with respect to the desired images and finally runs L-BFGS over the pixels of the generated image so as to minimize the sum of the weighted losses.

**Params.**

- `input_img`: tensor containing: content_img, style_img and output_img.
- `convnet`: [string], defines which VGG to use: vgg16 or vgg19.
- `style_layers`: list containing name of layers to use for style
  reconstruction. Defined in Gatys et. al but can be changed.
- `content_layer`: string containing name of layer to use for content
  reconstruction. Also defined in Gatys et. al.
- `content_weight`: weight for the content loss.
- `style_weight`: weight for the style loss.
- `tv_weight`: weight for the total variation loss.
- `iterations`: iterations for optimization algorithm
- `output_img_path`: path to output image.

The default content and style layers are:

```python
CONTENT_LAYER = 'block4_conv2'
STYLE_LAYERS = ['block1_conv1',
				'block2_conv1',
				'block3_conv1', 
				'block4_conv1', 
				'block5_conv1']
```
