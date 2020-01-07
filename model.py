import tensorflow as tf
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.keras.optimizer_v2.adam import Adam
import numpy as np
import PIL.Image
import IPython.display as display

"""
This code is slightly modified from TensorFlow's official tutorial (https://www.tensorflow.org/tutorials/generative/style_transfer), which implements
A Neural Algorithm of Artistic Style (Gatys et al.) (https://arxiv.org/abs/1508.06576)
"""

def tensor_to_image(tensor:resource_variable_ops=None):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img:str=None, dim:int=800):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def vgg_layers(layer_names:list=None):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor:resource_variable_ops=None):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers:list=None, content_layers:list=None):
        super(StyleContentModel, self).__init__()
        self.vgg =  vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content':content_dict, 'style':style_dict}
    
def clip_0_1(image:resource_variable_ops=None):
      return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    
def style_content_loss(outputs:dict=None,
                       style_weight:float=None,
                       content_weight:float=None,
                       num_style_layers:int=None,
                       num_content_layers:int=None,
                       style_targets:dict=None,
                       content_targets:dict=None):
    
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(image:resource_variable_ops=None,
               style_weight:float=1e-1,
               content_weight:float=1e3,
               num_style_layers:int=5,
               num_content_layers:int=1,
               style_targets:dict=None,
               content_targets:dict=None,
               extractor:dict=None,
               opt:Adam=None,
               total_variation_weight:int=30):
    
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_weight, content_weight, num_style_layers, num_content_layers, style_targets, content_targets)
        loss += total_variation_weight*tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

def train_model(epochs:int=30,
                steps_per_epoch:int=100,
                save_image:bool=False,
                img_name:str=None,
                display_image:bool=False,
                image:resource_variable_ops=None,
                style_weight:float=1e-1,
                content_weight:float=1e3,
                num_style_layers:int=5,
                num_content_layers:int=1,
                style_targets:dict=None,
                content_targets:dict=None,
                extractor:dict=None,
                opt:Adam=None,
                total_variation_weight:int=30):
    
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image, style_weight, content_weight, num_style_layers, num_content_layers, style_targets, content_targets, extractor, opt, total_variation_weight)
    
        if save_image:
            # Save image after every epoch
            tensor_to_image(image).save(f"{img_name}_{step}.png")
        
        if display_image:
            # Display image after every epoch
            display.clear_output(wait=True)
            display.display(tensor_to_image(image))
            
        print(f"Train step: {step}")

    return image

    