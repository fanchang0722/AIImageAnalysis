import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.keras import layers, models
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as ImagePrep
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained model
#model = tf.keras.applications.EfficientNetB0(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
#model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet')

# Modify the input shape of the model to match the grayscale image
#input_shape = (224, 224, 1)
#input_layer = tf.keras.layers.Input(shape=input_shape)
#model = tf.keras.Model(inputs=input_layer, outputs=model(input_layer))
# Modify the input shape of the model to match the grayscale image
#model._layers[0].batch_input_shape = (None, 224, 224, 1)

# Define the line coordinates
line_start = (178, 256)
line_end = (209, 225)

# Define the size of the crop region
crop_width = 224 #300
crop_height = 224 # 200

# Define the input and output directories
input_dir = './data/1 DNA repair/U2OS_53KO_5979GFP_FOV3'
output_dir = './prep/dna_repair/output'

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))

# Iterate over all TIFF files in the input directory
for filename in os.listdir(input_dir):
  if filename.endswith('.tif'):
    # Load the TIFF image
    input_path = os.path.join(input_dir, filename)
    img = Image.open(input_path)

    # Calculate the center coordinates of the line
    line_center = ((line_start[0] + line_end[0]) // 2, (line_start[1] + line_end[1]) // 2)

    # Calculate the left and right coordinates of the crop region
    left = line_center[0] - crop_width // 2
    right = line_center[0] + crop_width // 2

    # Calculate the top and bottom coordinates of the crop region
    top = line_center[1] - crop_height // 2
    bottom = line_center[1] + crop_height // 2

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))
    #   cv_img = cropped_img.resize((224, 224))
    #cv_img = np.expand_dims(cv_img, axis=0)
    #cv_img = cv_img.astype(np.float32) / 255.0
    x = ImagePrep.img_to_array(cropped_img)
    x = np.expand_dims(x, axis=0)

    # Convert the grayscale image to an RGB image with three channels
    x = np.concatenate([x, x, x], axis=-1)

    # Preprocess the input data
    x = preprocess_input(x)

    # Construct the output filename based on the input filename and the crop region coordinates
    basename, ext = os.path.splitext(filename)
    output_filename = f'{basename}_cropped_{left}_{top}_{right}_{bottom}.tif' #jpg'
    #output_path = os.path.join(output_dir, output_filename)
    print("predicting for: ", output_filename)

    # Show the original image in the first subplot
    ax1.imshow(img, cmap = "gray")
    ax1.set_title(basename)

    # Show the cropped image in the second subplot
    ax2.imshow(cropped_img, cmap = "gray")
    ax2.set_title(output_filename)
    
    # Display the figure
    #plt.show()

    
    # Use the model to predict the object classes and bounding boxes in the image
    preds = model.predict(x)
    #top_predictions = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=5)
    # Get the top 2 predicted class labels and probabilities
    top_preds = decode_predictions(preds, top=2)[0]
    # Print the top predictions
    #for prediction in top_predictions[0]:
    #    print('{}: {:.2f}%'.format(prediction[1], prediction[2]*100))
    print('Predicted:', top_preds)
    #cropped_img.mode = 'I'
    #cropped_img.point(lambda i:i*(1./256)).convert('L').save(output_path)
    # Save the cropped image as a JPEG file
    #cropped_img.save(output_path)

    # Get the top predicted class and its bounding box
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer('conv5_block3_out')
    #grads = tf.keras.backend.GradientTape(class_output, last_conv_layer.output)[0]
    grads = tf.GradientTape(class_output, last_conv_layer.output)[0]
    pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2))
    iterate = tf.keras.backend.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Resize the heatmap to the original image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(img.size)
    heatmap = np.array(heatmap)

    # Apply thresholding to the heatmap to get the object region
    threshold = 0.8
    heatmap_thresh = np.zeros_like(heatmap)
    heatmap_thresh[heatmap >= (threshold * 255)] = 1

    # Get the bounding box coordinates of the object region
    coords = np.argwhere(heatmap_thresh)
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    size_ratio = (x_max - x_min) / (y_max - y_min)

    # Draw the bounding box on the image
    draw = ImageDraw.Draw(cropped_img)
    draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)

    '''
    # Extract the bounding boxes for the top predicted class
    _, _, _, (xmin, ymin, xmax, ymax) = top_preds[0][2]

    # Scale the bounding box coordinates back to the original image size
    orig_size = cropped_img.size
    xmin = int(xmin * orig_size[0])
    ymin = int(ymin * orig_size[1])
    xmax = int(xmax * orig_size[0])
    ymax = int(ymax * orig_size[1])

    draw = ImageDraw.Draw(cropped_img)
    draw.rectangle((xmin, ymin, xmax, ymax), outline='red', width=3)
    '''
    # Show the image
    cropped_img.show()

'''

# Define the input shape of the images
input_shape = (224, 224, 3)

# Define the number of classes (objects) to detect
num_classes = 10

# Define the model architecture using transfer learning
base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
predictions = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('train', target_size=input_shape[:2], batch_size=32, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory('val', target_size=input_shape[:2], batch_size=32, class_mode='categorical')

# Train the model
model.fit_generator(train_generator, steps_per_epoch=100, epochs=10, validation_data=validation_generator, validation_steps=50)

# Fine-tune the model
for layer in base_model.layers:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=100, epochs=10, validation_data=validation_generator, validation_steps=50)
'''