import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, filters, segmentation
import tkinter as tk
from tkinter import filedialog

# Default threshold values
DEFAULT_LOW_THRESHOLD = 0.1
DEFAULT_HIGH_THRESHOLD = 0.065

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        ax1.image_path = file_path

def region_growing_segmentation(seed_point, original_image, ax, threshold):
    # Convert the image to grayscale
    grayscale_image = color.rgb2gray(original_image)
    
    # Create a binary mask for the region growing
    region_mask = np.zeros_like(grayscale_image, dtype=bool)
    
    # Set up the region growing parameters
    stack = [seed_point]
    visited = set()
    
    # Region growing algorithm
    while stack:
        x, y = stack.pop()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        if grayscale_image[x, y] < threshold:
            region_mask[x, y] = True
            if x > 0:
                stack.append((x - 1, y))
            if x < grayscale_image.shape[0] - 1:
                stack.append((x + 1, y))
            if y > 0:
                stack.append((x, y - 1))
            if y < grayscale_image.shape[1] - 1:
                stack.append((x, y + 1))
    
    # Apply the region mask to the original image
    segmented_image = np.zeros_like(grayscale_image)
    segmented_image[region_mask] = grayscale_image[region_mask]
    
    ax.imshow(segmented_image, cmap='gray')
    ax.axis('off')
    ax.set_title("Image segmentée")
    plt.show()

def perform_segmentation(seed_point, original_image, ax, threshold):
    # Perform region growing segmentation with the provided threshold
    region_growing_segmentation(seed_point, original_image, ax, threshold)

def update_segmentation(lower_a_scale, upper_a_scale, lower_b_scale, upper_b_scale, original_image, ax):
    lab_image = color.rgb2lab(original_image)
    a = lab_image[:, :, 1]
    b = lab_image[:, :, 2]
    lower_a = int(lower_a_scale.get())
    upper_a = int(upper_a_scale.get())
    lower_b = int(lower_b_scale.get())
    upper_b = int(upper_b_scale.get())
    mask = np.zeros_like(a, dtype=np.uint8)
    mask[(a >= lower_a) & (a <= upper_a) & (b >= lower_b) & (b <= upper_b)] = 255
    ax.imshow(mask, cmap='gray')
    ax.axis('off')
    ax.set_title("Image segmentée")
    plt.show()

def perform_watershed_segmentation(low_threshold_var, high_threshold_var):
    if ax1.image_path:
        # Load the original image
        original_image = io.imread(ax1.image_path)
        
        # Convert the image to grayscale
        grayscale_image = color.rgb2gray(original_image)
        
        # Apply a gradient to the image
        gradient_image = filters.sobel(grayscale_image)
        
        # Get threshold values from Scale widgets
        low_threshold = low_threshold_var.get()
        high_threshold = high_threshold_var.get()
        
        # Apply thresholding to obtain markers
        markers = np.zeros_like(grayscale_image)
        markers[gradient_image < low_threshold * gradient_image.max()] = 1
        markers[gradient_image > high_threshold * gradient_image.max()] = 2
        
        # Apply watershed algorithm
        segmented_image = segmentation.watershed(gradient_image, markers)
        
        # Display the segmented image
        ax.imshow(segmented_image, cmap='nipy_spectral')
        ax.axis('off')
        ax.set_title("Image segmentée")
        plt.show()

def create_segmentation_window(seg_type):
    if ax1.image_path:
        # Load and display the original image
        original_image = io.imread(ax1.image_path)
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.axis('off')
        plt.title("Image originale")
        
        # Matplotlib Figure for Image Display
        global ax
        ax = plt.subplot(1, 2, 2)
        ax.image_path = None  # Store the image path in ax for later use
        
        # Choose a seed point for region growing (e.g., center of the image)
        seed_point = (original_image.shape[0] // 2, original_image.shape[1] // 2)
        
        # Create a new window for segmentation
        segmentation_window = tk.Toplevel(root)
        
        if seg_type == 1:
            segmentation_window.title("Technique de Segmentation Région de croissance")
            
            # Frame for Region Growing Parameters
            parameters_frame = tk.Frame(segmentation_window)
            parameters_frame.pack(pady=10)
            
            # Threshold Scale
            threshold_var = tk.DoubleVar()
            threshold_var.set(0.3)  # Default threshold value
            threshold_label = tk.Label(parameters_frame, text="Seuil de Croissance de la région:")
            threshold_label.pack(side=tk.LEFT, padx=10)
            threshold_scale = tk.Scale(parameters_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=threshold_var)
            threshold_scale.pack(side=tk.LEFT, padx=10)
            
            # Segment Button
            segment_button = tk.Button(segmentation_window, text="Mettre à jour", command=lambda: perform_segmentation(seed_point, original_image, ax, threshold_var.get()))
            segment_button.pack(pady=10)
        
        elif seg_type == 2:
            segmentation_window.title("Technique de Segmentation 1")
            
            # Segmentation Parameters Frame
            parameters_frame = tk.Frame(segmentation_window)
            parameters_frame.pack()
            
            lower_a_label = tk.Label(parameters_frame, text="Lower a:")
            lower_a_label.grid(row=0, column=0)
            lower_a_scale = tk.Scale(parameters_frame, from_=0, to=100, orient=tk.HORIZONTAL)
            lower_a_scale.set(0)
            lower_a_scale.grid(row=0, column=1)
            
            upper_a_label = tk.Label(parameters_frame, text="Upper a:")
            upper_a_label.grid(row=1, column=0)
            upper_a_scale = tk.Scale(parameters_frame, from_=0, to=1000, orient=tk.HORIZONTAL)
            upper_a_scale.set(100)
            upper_a_scale.grid(row=1, column=1)
            
            lower_b_label = tk.Label(parameters_frame, text="Lower b:")
            lower_b_label.grid(row=2, column=0)
            lower_b_scale = tk.Scale(parameters_frame, from_=-255, to=0, orient=tk.HORIZONTAL)
            lower_b_scale.set(-130)
            lower_b_scale.grid(row=2, column=1)
            
            upper_b_label = tk.Label(parameters_frame, text="Upper b:")
            upper_b_label.grid(row=3, column=0)
            upper_b_scale = tk.Scale(parameters_frame, from_=-255, to=0, orient=tk.HORIZONTAL)
            upper_b_scale.set(-15)
            upper_b_scale.grid(row=3, column=1)
            
            update_button = tk.Button(parameters_frame, text="Mettre à jour", command=lambda: update_segmentation(lower_a_scale, upper_a_scale, lower_b_scale, upper_b_scale, original_image, ax))
            update_button.grid(row=4, columnspan=2)
        
        elif seg_type == 3:
            segmentation_window.title("Technique de Segmentation Watershed")
            
            # Threshold Parameters Frame
            parameters_frame = tk.Frame(segmentation_window)
            parameters_frame.pack(pady=10)
            
            # Low Threshold Scale
            low_threshold_var = tk.DoubleVar()
            low_threshold_var.set(DEFAULT_LOW_THRESHOLD)
            low_threshold_scale = tk.Scale(parameters_frame, from_=0, to=1, resolution=0.01, label="Low Threshold", variable=low_threshold_var, orient=tk.HORIZONTAL)
            low_threshold_scale.pack(side=tk.LEFT, padx=10)
            
            # High Threshold Scale
            high_threshold_var = tk.DoubleVar()
            high_threshold_var.set(DEFAULT_HIGH_THRESHOLD)
            high_threshold_scale = tk.Scale(parameters_frame, from_=0, to=1, resolution=0.01, label="High Threshold", variable=high_threshold_var, orient=tk.HORIZONTAL)
            high_threshold_scale.pack(side=tk.LEFT, padx=10)
            
            # Button to trigger segmentation
            segment_button = tk.Button(segmentation_window, text="Mettre à jour", command=lambda: perform_watershed_segmentation(low_threshold_var, high_threshold_var))
            segment_button.pack(pady=10)
        
        root.wait_window(segmentation_window)

# Create main application window
root = tk.Tk()
root.title("Segmentation d'image")

# Browse Image Button
browse_button = tk.Button(root, text="Sélectionner une image", command=browse_image)
browse_button.pack(pady=10)

# Segment Buttons
segment_button1 = tk.Button(root, text="Segmentation par région de croissance", command=lambda: create_segmentation_window(1))
segment_button1.pack(pady=10)

segment_button2 = tk.Button(root, text="Segmentation par seuillage", command=lambda: create_segmentation_window(2))
segment_button2.pack(pady=10)

segment_button3 = tk.Button(root, text="Segmentation par watershed", command=lambda: create_segmentation_window(3))
segment_button3.pack(pady=10)

# Matplotlib Figure for Image Display
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.image_path = None  # Store the image path in ax1 for later use

root.mainloop()
