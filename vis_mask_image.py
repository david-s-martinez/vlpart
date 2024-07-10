import matplotlib.pyplot as plt
import cv2
import numpy as np


im = cv2.imread('/home/qf/Documents/test_vlpart/image_20240705_181108.png')
mask = np.load('/home/qf/Documents/test_vlpart/mask_20240705_181108.npy')
mask = mask[1]

# Plot the color image with the mask overlay
plt.figure(figsize=(6, 6))

# Display the color image
plt.imshow(im)

# Display the mask with transparency (alpha)
plt.imshow(mask, cmap='jet', alpha=0.5)  # 'jet' colormap for the mask

# Set the title and remove axis ticks
plt.title('Color Image with Mask Overlay')
plt.axis('off')

plt.show()