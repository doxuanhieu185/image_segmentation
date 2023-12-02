import cv2
import functions
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, filedialog, Button, Label, Frame

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("800x600")  # Set the initial size of the window

        self.image_path = None
        self.image_label = Label(root, text="No image selected", font=("Helvetica", 12))
        self.image_label.pack(pady=10)

        self.select_button = Button(root, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        self.execute_button = Button(root, text="Execute", command=self.execute_processing)
        self.execute_button.pack(pady=10)

        self.image_frame = Frame(root)
        self.image_frame.pack(pady=10)

        # Create a placeholder for the Matplotlib figure
        self.canvas = None

    def select_image(self):
        self.image_path = filedialog.askopenfilename(title="Select an image file",
                                                      filetypes=[("Image files", "*.jpg;*.png;*.bmp")])

        if not self.image_path:
            self.image_label.config(text="No image selected")
        else:
            self.image_label.config(text=f"Selected Image: {self.image_path}")

    def execute_processing(self):
        if not self.image_path:
            print("No image selected. Please select an image first.")
            return

        image1 = cv2.imread(self.image_path)

        # Call all the functions
        resized_image = functions.image_resize_dimensions(image1, ratio=0.5)
        threshold_result = functions.image_threshold(image1, threshold=100, adaptive=False, otsu=False)
        sobel_result = functions.image_sobel_gradient(image1)
        canny_edges_result = functions.image_canny_edges(image1)
        kmean_segmentation_result = functions.image_kmean_segmentation(image1, k=3)

        # Display the results in the Tkinter window
        self.display_results(resized_image, threshold_result, sobel_result, canny_edges_result, kmean_segmentation_result)

    def display_results(self, resized_image, threshold_result, sobel_result, canny_edges_result, kmean_segmentation_result):
        # Clear the existing Matplotlib figure
        plt.clf()

        plt.figure(figsize=(12, 8))

        plt.subplot(231), plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
        plt.subplot(232), plt.imshow(canny_edges_result, cmap='gray'), plt.title('Canny Edges Result')
        plt.subplot(233), plt.imshow(threshold_result, cmap='gray'), plt.title('Threshold Result')
        plt.subplot(234), plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)), plt.title('Grayscale Image')
        plt.subplot(235), plt.imshow(cv2.cvtColor(kmean_segmentation_result, cv2.COLOR_BGR2RGB)), plt.title('K-Means Segmentation Result')
        plt.subplot(236), plt.imshow(sobel_result, cmap='gray'), plt.title('Sobel Result')

        # Update the Matplotlib figure in the Tkinter window
        self.update_figure()

    def update_figure(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.image_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=1)

        plt.tight_layout()

if __name__ == "__main__":
    root = Tk()
    app = ImageProcessor(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)  # Handle the window close event
    root.mainloop()
