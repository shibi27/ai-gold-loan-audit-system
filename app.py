import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

# Use TkAgg for embedding matplotlib figures into Tkinter
matplotlib.use('TkAgg')

class GoldChainAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("AI powered Gold Loan Audit System")
        self.root.geometry("1400x900")
        
        # Variables to store images
        self.original_image = None
        self.background_removed_image = None
        self.contour_image = None
        self.advanced_contour_image = None
        self.comparison_image1 = None
        self.comparison_image2 = None
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.analysis_tab = ttk.Frame(self.notebook)
        self.comparison_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.analysis_tab, text='Chain Analysis')
        self.notebook.add(self.comparison_tab, text='Image Comparison')
        
        self.setup_analysis_tab()
        self.setup_comparison_tab()
    
    def setup_analysis_tab(self):
        """Setup the simplified chain analysis tab"""
        # Main frame
        main_frame = ttk.Frame(self.analysis_tab, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
        # Configure grid weights for resizing
        self.analysis_tab.columnconfigure(0, weight=1)
        self.analysis_tab.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
    
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="AI powered Gold Loan Audit System", 
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
    
        # Description
        desc_label = ttk.Label(
            main_frame,
            text="Upload an image of a gold chain to analyze length, composition (gold/stone percentage), and detect contours.",
            wraplength=1200
        )
        desc_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
    
        # Upload button
        upload_btn = ttk.Button(
            main_frame,
            text="Upload Image",
            command=self.upload_image
        )
        upload_btn.grid(row=2, column=0, pady=(0, 10), sticky=tk.W)
    
        # Analyze buttons
        self.analyze_btn = ttk.Button(
            main_frame,
            text="Analyze Chain",
            command=self.analyze_chain,
            state="disabled"
        )
        self.analyze_btn.grid(row=2, column=1, pady=(0, 10), sticky=tk.W)

        self.analyze_length_btn = ttk.Button(
            main_frame,
            text="Analyze Length",
            command=self.analyze_length,
            state="disabled"
        )
        self.analyze_length_btn.grid(row=2, column=2, pady=(0, 10), sticky=tk.W)
    
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(0, weight=1)
    
        # Original image
        original_frame = ttk.LabelFrame(results_frame, text="Original Image")
        original_frame.grid(row=0, column=0, padx=(0, 10), pady=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)
    
        self.original_label = ttk.Label(original_frame, text="No image uploaded")
        self.original_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
        # Final result with background removed and contours
        final_frame = ttk.LabelFrame(results_frame, text="Final Result (Background Removed with Contours)")
        final_frame.grid(row=0, column=1, padx=(10, 0), pady=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        final_frame.columnconfigure(0, weight=1)
        final_frame.rowconfigure(0, weight=1)
    
        self.final_label = ttk.Label(final_frame, text="Final result will appear here")
        self.final_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        # Length section
        length_frame = ttk.LabelFrame(results_frame, text="Length Analysis")
        length_frame.grid(row=0, column=2, padx=(10, 0), pady=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        length_frame.columnconfigure(0, weight=1)
        length_frame.rowconfigure(0, weight=1)
        length_frame.rowconfigure(1, weight=0)

        # Dynamic length label inside the frame
        self.length_text_var = tk.StringVar()
        self.length_text_var.set("Length: Not Calculated")
        length_info_label = ttk.Label(length_frame, textvariable=self.length_text_var, font=("Arial", 12, "bold"))
        length_info_label.grid(row=0, column=0, pady=(0, 10))

        # Image output (skeleton visualization)
        self.length_image_label = ttk.Label(length_frame, text="Analysis result will appear here")
        self.length_image_label.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

    
        self.pie_chart_frame = ttk.Frame(length_frame)
        self.pie_chart_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to upload image")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken")
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(20, 0))

    
    def setup_comparison_tab(self):
        """Setup the image comparison tab"""
        main_frame = ttk.Frame(self.comparison_tab, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.comparison_tab.columnconfigure(0, weight=1)
        self.comparison_tab.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Image Similarity Analysis", 
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Description
        desc_label = ttk.Label(
            main_frame,
            text="Upload two images to compare their similarity using multiple feature matching algorithms.",
            wraplength=1200
        )
        desc_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # Upload buttons
        upload_frame = ttk.Frame(main_frame)
        upload_frame.grid(row=2, column=0, columnspan=3, pady=(0, 20), sticky=(tk.W, tk.E))
        
        ttk.Button(upload_frame, text="Upload Image 1", 
                  command=lambda: self.upload_comparison_image(1)).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(upload_frame, text="Upload Image 2", 
                  command=lambda: self.upload_comparison_image(2)).grid(row=0, column=1, padx=(10, 0))
        
        # Compare button
        self.compare_btn = ttk.Button(
            main_frame,
            text="Compare Images",
            command=self.compare_images,
            state="disabled"
        )
        self.compare_btn.grid(row=3, column=0, columnspan=3, pady=(0, 20))
        
        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Comparison Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.columnconfigure(1, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Original images (with background removed)
        images_frame = ttk.Frame(results_frame)
        images_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        
        img1_frame = ttk.LabelFrame(images_frame, text="Image 1 (Background Removed)")
        img1_frame.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E, tk.N, tk.S))
        self.comparison_label1 = ttk.Label(img1_frame, text="No image uploaded")
        self.comparison_label1.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        img2_frame = ttk.LabelFrame(images_frame, text="Image 2 (Background Removed)")
        img2_frame.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        self.comparison_label2 = ttk.Label(img2_frame, text="No image uploaded")
        self.comparison_label2.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Best match result (with background removed)
        best_match_frame = ttk.LabelFrame(results_frame, text="Best Matching Result (Background Removed)")
        best_match_frame.grid(row=1, column=0, padx=(0, 5), pady=(10, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        best_match_frame.columnconfigure(0, weight=1)
        best_match_frame.rowconfigure(0, weight=1)
        
        self.best_match_label = ttk.Label(best_match_frame, text="Matching result will appear here")
        self.best_match_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results table
        table_frame = ttk.LabelFrame(results_frame, text="Detailed Comparison Results")
        table_frame.grid(row=1, column=1, padx=(5, 0), pady=(10, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # Create treeview for results table
        columns = ("Method", "Keypoints", "Initial Matches", "RANSAC Matches", "Inlier Ratio", "Confidence", "Similarity", "Status")
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        
        # Configure columns
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=110)
        
        # Add scrollbar
        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Comparison status
        self.comparison_status = tk.StringVar()
        self.comparison_status.set("Upload two images to compare")
        comparison_status_bar = ttk.Label(main_frame, textvariable=self.comparison_status, relief="sunken")
        comparison_status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.status_var.set("Loading image...")
                self.root.update()
                
                # Load and display original image
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("Could not load image from file")
                
                # Convert BGR to RGB for display
                original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.display_image(original_rgb, self.original_label)
                
                # Enable analyze button
                self.analyze_btn.config(state="normal")
                self.analyze_length_btn.config(state="normal")
                self.status_var.set("Image loaded successfully. Click 'Analyze Chain' or 'Analyze Length' to process.")

                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_var.set("Error loading image")
    
    def analyze_length(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Upload an image first.")
            return

        try:
            self.status_var.set("Analyzing chain length...")
            self.root.update_idletasks()

            img = self.original_image.copy()
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Detect gold-like color
            lower_gold = np.array([10, 40, 40])
            upper_gold = np.array([45, 255, 255])
            mask = cv2.inRange(hsv, lower_gold, upper_gold)

            # Clean up noise
            mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
            _, binary_mask = cv2.threshold(mask_blur, 50, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

            # Skeletonize for length
            from skimage.morphology import skeletonize
            skeleton = skeletonize(binary_mask > 0)
            length_pixels = int(np.sum(skeleton))

            # ✅ Update UI labels
            self.length_text_var.set(f"Length: {length_pixels} px")
            self.status_var.set(f"Necklace Length: {length_pixels} pixels")

            # ✅ Overlay skeleton on image
            skeleton_vis = np.zeros_like(img)
            skeleton_vis[skeleton] = [0, 255, 0]  # bright green
            overlay = cv2.addWeighted(img, 0.7, skeleton_vis, 1, 0)

            # Add length text directly on overlay
            cv2.putText(
                overlay,
                f"Length: {length_pixels} px",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # ✅ Ensure valid dimensions
            if overlay is None or overlay.size == 0:
                raise ValueError("Overlay image not generated properly")

            # ✅ Convert BGR → RGB for Tkinter
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            # ✅ Display image clearly in the Length section
            self.display_image(overlay_rgb, self.length_image_label, size=400)
            self.root.update_idletasks()

        except Exception as e:
            messagebox.showerror("Error", f"Length analysis failed: {str(e)}")
            self.status_var.set("Error during length analysis")


    def analyze_chain(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return

        try:
            self.status_var.set("Processing image with advanced contour detection and composition analysis...")
            self.root.update()

            final_result, analysis_info, contour_length, gold_percent, stone_percent = self.advanced_contour_detection_with_composition(self.original_image)

            if final_result is not None:
                original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.display_image(original_rgb, self.original_label)
                self.display_image(final_result, self.final_label)

                analysis_image = self.create_analysis_image_with_composition(final_result, contour_length, gold_percent, stone_percent)
                self.display_image(analysis_image, self.length_image_label, size=350)

                self.create_pie_chart(gold_percent, stone_percent)

                self.status_var.set("Analysis completed successfully")
            else:
                messagebox.showerror("Error", "Failed to process image with contour detection")
                self.status_var.set("Error during processing")

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_var.set("Error during processing")


    def advanced_contour_detection_with_composition(self, image: np.ndarray):
        try:
            original_img = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 90)
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            object_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 20]
            if not large_contours:
                return None, "No contours found.", 0, 0, 0

            cv2.drawContours(object_mask, large_contours, -1, 255, -1)
            mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            h, w = mask.shape
            mask_floodfill = mask.copy()
            temp_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(mask_floodfill, temp_mask, (0, 0), 0)
            final_mask = mask_floodfill

            hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
            lower_gold, upper_gold = np.array([10, 40, 40]), np.array([45, 255, 255])
            gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
            lower_stone, upper_stone = np.array([0, 0, 0]), np.array([180, 255, 255])
            stone_mask = cv2.inRange(hsv, lower_stone, upper_stone)
            gold_refined_mask = cv2.bitwise_and(final_mask, gold_mask)
            stone_refined_mask = cv2.bitwise_and(final_mask, stone_mask)
            stone_refined_mask = cv2.bitwise_and(stone_refined_mask, cv2.bitwise_not(gold_refined_mask))
            gold_area = np.count_nonzero(gold_refined_mask)
            stone_area = np.count_nonzero(stone_refined_mask)
            total_area = gold_area + stone_area
            gold_percent = (gold_area / total_area) * 100 if total_area > 0 else 0
            stone_percent = (stone_area / total_area) * 100 if total_area > 0 else 0

            background_removed = cv2.bitwise_and(original_img, original_img, mask=final_mask)
            final_result = cv2.add(background_removed, np.zeros_like(original_img))
            cv2.drawContours(final_result, large_contours, -1, (0, 255, 0), 3)
            contour_length = sum(cv2.arcLength(cnt, True) for cnt in large_contours)
            return final_result, "Contours and composition analyzed", contour_length, gold_percent, stone_percent

        except Exception as e:
            print(f"Error in advanced contour detection: {e}")
            return None, f"Error: {str(e)}", 0, 0, 0


    def upload_comparison_image(self, image_num):
        file_path = filedialog.askopenfilename(
            title=f"Select Image {image_num}",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Could not load image from file")
                
                # Remove background for comparison images
                bg_removed_image, _ = self.remove_background_for_comparison(image)
                
                if image_num == 1:
                    self.comparison_image1 = image
                    self.display_image(bg_removed_image, self.comparison_label1, size=300)
                else:
                    self.comparison_image2 = image
                    self.display_image(bg_removed_image, self.comparison_label2, size=300)
                
                # Enable compare button if both images are loaded
                if self.comparison_image1 is not None and self.comparison_image2 is not None:
                    self.compare_btn.config(state="normal")
                    self.comparison_status.set("Both images loaded. Click 'Compare Images' to analyze similarity.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def create_analysis_image_with_composition(self, image, length, gold_percent, stone_percent):
        """Create an image with length and composition information overlaid on top"""
        # Convert numpy array to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        # Create a copy to draw on
        draw_image = pil_image.convert('RGBA')
        draw = ImageDraw.Draw(draw_image)
        
        try:
            # Try to use a larger font
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_medium = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            # Fallback to default font if arial is not available
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
        
        # Get image dimensions
        width, height = draw_image.size
        
        # Create a semi-transparent background for text
        text_bg_height = 120
        text_bg = Image.new('RGBA', (width, text_bg_height), (0, 0, 0, 200))
        draw_image.paste(text_bg, (0, 0), text_bg)
        
        # Add length information at the top
        length_text = f"Chain Length: {length:.2f} pixels"
        draw.text((width//2, 15), length_text, fill=(255, 255, 255, 255), font=font_large, anchor="mm")
        
        # Add composition information
        gold_text = f"Gold: {gold_percent:.1f}%"
        stone_text = f"Stone: {stone_percent:.1f}%"
        
        draw.text((width//2, 45), gold_text, fill=(255, 215, 0, 255), font=font_medium, anchor="mm")  # Gold color
        draw.text((width//2, 70), stone_text, fill=(192, 192, 192, 255), font=font_medium, anchor="mm")  # Silver color
        
        # Add total
        total_text = f"Total: 100.0%"
        draw.text((width//2, 95), total_text, fill=(255, 255, 255, 255), font=font_medium, anchor="mm")
        
        # Convert back to numpy array (BGR)
        return cv2.cvtColor(np.array(draw_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    def create_pie_chart(self, gold_percent, stone_percent):
        """Create a pie chart showing gold and stone composition"""
        # Clear previous pie chart
        for widget in self.pie_chart_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(4, 3))
        
        # Data for pie chart
        labels = [f'Gold: {gold_percent:.1f}%', f'Stone: {stone_percent:.1f}%']
        sizes = [gold_percent, stone_percent]
        colors = ['gold', 'lightgray']
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                         startangle=90)
        
        # Style the text
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        ax.set_title('Composition Analysis')
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, self.pie_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)
    
    def compare_images(self):
        if self.comparison_image1 is None or self.comparison_image2 is None:
            messagebox.showwarning("Warning", "Please upload both images first.")
            return
        
        try:
            self.comparison_status.set("Comparing images...")
            self.root.update()
            
            # Clear previous results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Remove background from both images for comparison
            img1_bg_removed, _ = self.remove_background_for_comparison(self.comparison_image1)
            img2_bg_removed, _ = self.remove_background_for_comparison(self.comparison_image2)
            
            # Perform comparison using all methods
            methods = ["SIFT", "ORB", "BRISK", "AKAZE", "KAZE"]
            results = []
            
            for method in methods:
                result = self.match_images(img1_bg_removed, img2_bg_removed, method)
                results.append(result)
                
                # Add to treeview
                self.results_tree.insert("", "end", values=(
                    result["Method"],
                    result["Keypoints"],
                    result["Initial Matches"],
                    result["RANSAC Matches"],
                    result["Inlier Ratio"],
                    result["Confidence Score"],
                    result["Similarity Score"],
                    result["Status"]
                ))
            
            # Find best result (by similarity numeric value)
            def sim_value(r):
                try:
                    return float(r['Similarity Score'].strip('%'))
                except Exception:
                    return 0.0
            
            best_result = max(results, key=sim_value)
            
            # Display best match result (already background removed)
            self.display_image(best_result["Matched Image"], self.best_match_label, size=400)
            
            # Calculate SSIM on background-removed images
            ssim_score = self.whole_chain_similarity(img1_bg_removed, img2_bg_removed)
            
            self.comparison_status.set(
                f"Comparison completed. Best method: {best_result['Method']} "
                f"(Similarity: {best_result['Similarity Score']}, SSIM: {ssim_score}%)"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Comparison failed: {str(e)}")
            self.comparison_status.set("Error during comparison")
    
    def display_image(self, image_array, label_widget, size=400):
        """Convert numpy array to PhotoImage and display in label"""
        # Remove any text placeholder
        label_widget.configure(text='')
        
        # Handle single-channel images
        if len(image_array.shape) == 2:
            pil_image = Image.fromarray(image_array)
            h, w = image_array.shape
        else:
            h, w = image_array.shape[:2]
            # If image is BGR (OpenCV), convert to RGB
            if image_array.shape[2] == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
            elif image_array.shape[2] == 4:
                # Assume RGBA
                pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGBA))
            else:
                pil_image = Image.fromarray(image_array)
        
        # Preserve aspect ratio when resizing
        if h > w:
            new_h = size
            new_w = int(w * (size / h))
        else:
            new_w = size
            new_h = int(h * (size / w))
        
        pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        label_widget.configure(image=photo)
        label_widget.image = photo
    
    def remove_background_for_comparison(self, image: np.ndarray):
        """
        Remove background for comparison images (simplified version without contours)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use Canny edge detection
            edges = cv2.Canny(blurred, 30, 90)
            
            # Dilate the edges to close small gaps
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create an empty mask for the object
            object_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Filter contours by area to remove small noise
            min_contour_area = 100
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
            
            if not large_contours:
                # Return original image if no contours found
                return image, "No significant contours found."
            
            # Draw large contours onto the object mask
            cv2.drawContours(object_mask, large_contours, -1, 255, -1)
            
            # Apply morphological operations to refine the mask
            mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Use flood fill to remove the background outside the object
            h, w = mask.shape
            mask_floodfill = mask.copy()
            temp_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(mask_floodfill, temp_mask, (0, 0), 0)
            
            # final_mask is the mask of foreground
            final_mask = mask_floodfill
            
            # Create background-removed image (foreground on black)
            background_removed = cv2.bitwise_and(image, image, mask=final_mask)
            black_background = np.zeros_like(image)
            final_with_black_bg = cv2.add(background_removed, black_background)
            
            return final_with_black_bg, "Background removed successfully"
                        
        except Exception as e:
            print(f"Error in background removal for comparison: {e}")
            return image, f"Error during background removal: {str(e)}"
    
    def advanced_contour_detection_with_composition(self, image: np.ndarray):
        """
        Advanced contour detection with gold/stone composition analysis
        """
        try:
            # Keep a copy of the original image
            original_img = image.copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use Canny edge detection
            edges = cv2.Canny(blurred, 30, 90)
            
            # Dilate the edges to close small gaps
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours
            contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create an empty mask for the object
            object_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Filter contours by area to remove small noise
            min_contour_area = 100
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
            
            if not large_contours:
                return None, "No significant contours found.", 0, 0, 0
            
            # Draw large contours onto the object mask
            cv2.drawContours(object_mask, large_contours, -1, 255, -1)
            
            # Apply morphological operations to refine the mask
            mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Use flood fill to remove the background outside the object
            h, w = mask.shape
            mask_floodfill = mask.copy()
            temp_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(mask_floodfill, temp_mask, (0, 0), 0)
            
            # final_mask is the mask of foreground
            final_mask = mask_floodfill
            
            # --- Gold and Stone Composition Analysis ---
            hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
            
            # Gold color range in HSV (tuned broadly)
            lower_gold = np.array([10, 40, 40])
            upper_gold = np.array([45, 255, 255])
            gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
            
            # Stone color range (everything else in the object)
            lower_stone = np.array([0, 0, 0])
            upper_stone = np.array([180, 255, 255])
            stone_mask = cv2.inRange(hsv, lower_stone, upper_stone)
            
            # Combine color masks with the object mask to refine
            gold_refined_mask = cv2.bitwise_and(final_mask, gold_mask)
            stone_refined_mask = cv2.bitwise_and(final_mask, stone_mask)
            
            # Remove overlap: areas that are both gold and stone
            stone_refined_mask = cv2.bitwise_and(stone_refined_mask, cv2.bitwise_not(gold_refined_mask))
            
            # Calculate percentages
            gold_area = np.count_nonzero(gold_refined_mask)
            stone_area = np.count_nonzero(stone_refined_mask)
            total_area = gold_area + stone_area
            
            if total_area == 0:
                gold_percent = 0.0
                stone_percent = 0.0
            else:
                gold_percent = (gold_area / total_area) * 100.0
                stone_percent = (stone_area / total_area) * 100.0
            
            # Find contours on the final mask for drawing
            all_contours, _ = cv2.findContours(final_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create background-removed image
            background_removed = cv2.bitwise_and(original_img, original_img, mask=final_mask)
            
            # Create black background
            black_background = np.zeros_like(original_img)
            
            # Combine foreground with black background
            final_with_black_bg = cv2.add(background_removed, black_background)
            
            # Draw contours on the background-removed image
            final_result = final_with_black_bg.copy()
            cv2.drawContours(final_result, all_contours, -1, (0, 255, 0), 3)  # Green contours, thickness 3
            
            # Calculate contour analysis
            analysis_info, contour_length = self.calculate_contour_analysis(all_contours, final_mask)
            
            # Add composition info to analysis
            analysis_info += f"\nCOMPOSITION ANALYSIS:\n"
            analysis_info += f"Gold Area: {gold_percent:.2f}%\n"
            analysis_info += f"Stone Area: {stone_percent:.2f}%\n"
            analysis_info += f"Total Analyzed Area: {total_area} pixels\n"
            
            return final_result, analysis_info, contour_length, gold_percent, stone_percent
                        
        except Exception as e:
            print(f"Error in advanced contour detection: {e}")
            return None, f"Error during processing: {str(e)}", 0, 0, 0
        
    def calculate_contour_analysis(self, contours, mask):
        """Calculate detailed contour analysis and return both text and length"""
        analysis_info = "=== Chain Length Analysis ===\n\n"
        
        if not contours:
            analysis_info += "No contours found.\n"
            return analysis_info, 0
        
        # Find the largest contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Basic contour properties
        contour_length = cv2.arcLength(main_contour, closed=False)
        contour_area = cv2.contourArea(main_contour)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Convex hull
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0
        
        analysis_info += f"MAIN CONTOUR PROPERTIES:\n"
        analysis_info += f"Chain Length: {contour_length:.2f} pixels\n"
        analysis_info += f"Contour Area: {contour_area:.2f} square pixels\n"
        analysis_info += f"Bounding Box: {w} x {h} pixels\n"
        analysis_info += f"Aspect Ratio: {aspect_ratio:.2f}\n"
        analysis_info += f"Solidity: {solidity:.3f}\n"
        analysis_info += f"Contour Points: {len(main_contour)}\n\n"
        
        analysis_info += f"DETECTION SUMMARY:\n"
        analysis_info += f"Total contours found: {len(contours)}\n"
        analysis_info += f"Foreground pixels: {np.sum(mask > 0)}\n"
        analysis_info += f"Background pixels: {np.sum(mask == 0)}\n"
        
        return analysis_info, contour_length
    
    # --- Image Comparison Functions ---
    
    def whole_chain_similarity(self, img1, img2):
        """Calculate structural similarity between two images"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Resize for fair comparison
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

        score, _ = ssim(gray1, gray2, full=True)
        return round(score * 100, 2)
    
    def match_images(self, img1, img2, method="SIFT"):
        """Match two images using feature detection algorithms"""
        try:
            if method == "SIFT":
                detector = cv2.SIFT_create()
            elif method == "ORB":
                detector = cv2.ORB_create()
            elif method == "BRISK":
                detector = cv2.BRISK_create()
            elif method == "AKAZE":
                detector = cv2.AKAZE_create()
            elif method == "KAZE":
                detector = cv2.KAZE_create()
            else:
                raise ValueError(f"Unsupported method: {method}")

            kp1, des1 = detector.detectAndCompute(img1, None)
            kp2, des2 = detector.detectAndCompute(img2, None)

            if des1 is None or des2 is None:
                return {
                    "Method": method,
                    "Keypoints": f"{len(kp1)}/{len(kp2)}",
                    "Initial Matches": 0,
                    "RANSAC Matches": 0,
                    "Inlier Ratio": 0,
                    "Confidence Score": "0%",
                    "Similarity Score": "0.00%",
                    "Status": "Not Similar",
                    "Matched Image": img1
                }

            if method in ["SIFT", "KAZE", "AKAZE"]:
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            else:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

            # KNN match (k=2)
            matches = matcher.knnMatch(des1, des2, k=2)

            good = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            initial_matches = len(matches)

            if len(good) > 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist() if mask is not None else [1]*len(good)
            else:
                matchesMask = [1]*len(good)

            inliers = sum(matchesMask) if matchesMask else 0
            inlier_ratio = inliers / max(len(good), 1)
            confidence = round(inlier_ratio * 100, 2)

            # Similarity score based on percentage of good matches
            total_keypoints = len(kp1)
            if total_keypoints > 0:
                similarity_score = min(100.0, round((len(good) / total_keypoints) * 100.0, 2))
            else:
                similarity_score = 0.0

            status = "Similar" if similarity_score > 5 else "Not Similar"

            if good:
                matched_img = cv2.drawMatches(
                    img1, kp1, img2, kp2, good, None,
                    matchColor=(0, 255, 0), singlePointColor=None,
                    matchesMask=None, flags=2
                )
            else:
                matched_img = img1.copy()

            return {
                "Method": method,
                "Keypoints": f"{len(kp1)}/{len(kp2)}",
                "Initial Matches": initial_matches,
                "RANSAC Matches": inliers,
                "Inlier Ratio": round(inlier_ratio, 2),
                "Confidence Score": f"{confidence}%",
                "Similarity Score": f"{similarity_score:.2f}%",
                "Status": status,
                "Matched Image": matched_img
            }
        except Exception as e:
            print(f"Error in match_images ({method}): {e}")
            return {
                "Method": method,
                "Keypoints": "0/0",
                "Initial Matches": 0,
                "RANSAC Matches": 0,
                "Inlier Ratio": 0,
                "Confidence Score": "0%",
                "Similarity Score": "0.00%",
                "Status": "Error",
                "Matched Image": img1 if img1 is not None else np.zeros((100,100,3), dtype=np.uint8)
            }

def main():
    root = tk.Tk()
    app = GoldChainAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
