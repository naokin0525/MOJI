# src/ui/gui.py
"""
Tkinter-based GUI application for the Handwriting Generator.

Allows users to select models, enter text, choose styles, generate handwriting,
and preview the output.

Requires: `pip install Pillow cairosvg`
May also require Cairo system library.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import logging
import io
import time

# --- Try importing required libraries ---
try:
    from PIL import Image, ImageTk
except ImportError:
    messagebox.showerror("Missing Dependency", "Pillow library not found.\nPlease install it: pip install Pillow")
    exit()
try:
    import cairosvg
except ImportError:
     messagebox.showerror("Missing Dependency", "cairosvg library not found.\nPlease install it: pip install cairosvg\n(May also require system Cairo library)")
     exit()

# --- Try importing project modules ---
# Use relative imports assuming standard execution from project root or installed package
try:
    from ..generation.generator import HandwritingGenerator
    from ..generation.style_control import STYLE_PRESETS
    from ..models.vaegan import HandwritingVAEGAN # Needed for loading model
    from ..utils.error_handling import ModelLoadError, GenerationError, ConversionError
except ImportError as e:
    # Fallback for running script directly during development (adjust paths if needed)
    try:
        import sys
        # Add project root to path temporarily if running script directly
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
             sys.path.insert(0, project_root)
        from src.generation.generator import HandwritingGenerator
        from src.generation.style_control import STYLE_PRESETS
        from src.models.vaegan import HandwritingVAEGAN
        from src.utils.error_handling import ModelLoadError, GenerationError, ConversionError
    except ImportError:
         messagebox.showerror("Import Error", f"Could not import necessary project modules: {e}.\nEnsure the script is run correctly within the project structure.")
         exit()

import torch # Import torch here after ensuring project path is set if needed

logger = logging.getLogger(__name__)
# Configure basic logging for the GUI if not configured upstream
if not logging.getLogger().hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')


class HandwritingApp(tk.Tk):
    """Main application class for the Handwriting Generation GUI."""

    def __init__(self):
        super().__init__()
        self.title("AI Handwriting Generator")
        self.geometry("800x600") # Initial size

        # --- Data Variables ---
        self.model_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=os.path.join(os.getcwd(), "output_gui"))
        self.style_var = tk.StringVar(value='default')
        self.stroke_width_var = tk.DoubleVar(value=1.0)
        self.random_variation_scale_var = tk.DoubleVar(value=1.0) # Scale for sampling noise
        self.seed_var = tk.StringVar() # Use string var to allow empty input
        self.output_format_var = tk.StringVar(value='svg')

        # Internal state
        self.loaded_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preview_image_tk = None # To hold reference to PhotoImage
        self.generator_thread = None

        # --- Layout ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3) # Preview area larger
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0) # Status bar fixed height

        # --- Create Widgets ---
        self._create_control_widgets()
        self._create_preview_area()
        self._create_status_bar()

        logger.info(f"Handwriting GUI initialized. Using device: {self.device}")

    def _create_control_widgets(self):
        """Creates the left-side panel with all input controls."""
        control_frame = ttk.Frame(self, padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        row = 0

        # Model Selection
        ttk.Label(control_frame, text="Model File (.pth):").grid(row=row, column=0, sticky="w", pady=2)
        model_entry = ttk.Entry(control_frame, textvariable=self.model_path_var, state='readonly', width=40)
        model_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        model_button = ttk.Button(control_frame, text="Browse...", command=self._browse_model_file)
        model_button.grid(row=row, column=2, sticky="e", padx=5, pady=2)
        row += 1

        # Text Input
        ttk.Label(control_frame, text="Input Text:").grid(row=row, column=0, sticky="nw", pady=2)
        self.text_input = tk.Text(control_frame, height=8, width=40, wrap=tk.WORD)
        self.text_input.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=2)
        row += 1

        # Style Selection
        ttk.Label(control_frame, text="Style:").grid(row=row, column=0, sticky="w", pady=2)
        style_options = sorted(list(STYLE_PRESETS.keys()))
        self.style_combo = ttk.Combobox(control_frame, textvariable=self.style_var, values=style_options, state='readonly', width=15)
        self.style_combo.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        row += 1

        # Parameters Frame
        param_frame = ttk.LabelFrame(control_frame, text="Parameters", padding="5")
        param_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=10)
        param_row=0
        # Stroke Width
        ttk.Label(param_frame, text="Stroke Width:").grid(row=param_row, column=0, sticky="w", pady=2)
        ttk.Scale(param_frame, from_=0.1, to=5.0, variable=self.stroke_width_var, orient=tk.HORIZONTAL).grid(row=param_row, column=1, sticky="ew", padx=5)
        ttk.Label(param_frame, textvariable=self.stroke_width_var, width=4).grid(row=param_row, column=2, sticky="w") # Show value
        param_row += 1
        # Random Variation
        ttk.Label(param_frame, text="Random Variation:").grid(row=param_row, column=0, sticky="w", pady=2)
        ttk.Scale(param_frame, from_=0.0, to=2.0, variable=self.random_variation_scale_var, orient=tk.HORIZONTAL).grid(row=param_row, column=1, sticky="ew", padx=5)
        ttk.Label(param_frame, textvariable=self.random_variation_scale_var, width=4).grid(row=param_row, column=2, sticky="w") # Show value
        param_row += 1
        # Seed
        ttk.Label(param_frame, text="Seed (optional):").grid(row=param_row, column=0, sticky="w", pady=2)
        ttk.Entry(param_frame, textvariable=self.seed_var, width=10).grid(row=param_row, column=1, sticky="w", padx=5)
        param_row += 1
        param_frame.columnconfigure(1, weight=1) # Make scale expand
        row += 1

        # Output Options
        out_frame = ttk.LabelFrame(control_frame, text="Output", padding="5")
        out_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        out_row = 0
        # Output Directory
        ttk.Label(out_frame, text="Output Directory:").grid(row=out_row, column=0, sticky="w", pady=2)
        out_dir_entry = ttk.Entry(out_frame, textvariable=self.output_dir_var, state='readonly', width=30)
        out_dir_entry.grid(row=out_row, column=1, sticky="ew", padx=5, pady=2)
        out_dir_button = ttk.Button(out_frame, text="Browse...", command=self._browse_output_dir)
        out_dir_button.grid(row=out_row, column=2, sticky="e", padx=5, pady=2)
        out_row += 1
        # Output Format
        ttk.Label(out_frame, text="Format:").grid(row=out_row, column=0, sticky="w", pady=2)
        format_combo = ttk.Combobox(out_frame, textvariable=self.output_format_var, values=['svg', 'png', 'jpg'], state='readonly', width=5)
        format_combo.grid(row=out_row, column=1, sticky="w", padx=5, pady=2)
        out_row += 1
        out_frame.columnconfigure(1, weight=1) # Make entry expand
        row += 1

        # Generate Button
        self.generate_button = ttk.Button(control_frame, text="Generate Handwriting", command=self._start_generation, style='Accent.TButton') # Requires themed style if available
        self.generate_button.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=15)
        row += 1

        # Allow control frame columns to resize
        control_frame.columnconfigure(1, weight=1)

    def _create_preview_area(self):
        """Creates the right-side panel for image preview."""
        preview_frame = ttk.LabelFrame(self, text="Preview", padding="10")
        preview_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        # Use a Label to display the image
        self.preview_label = ttk.Label(preview_frame, background="lightgrey", anchor=tk.CENTER)
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        # Could use a Canvas if drawing/zooming needed later

        # Keep track of label size for resizing image
        self.preview_label.bind("<Configure>", self._on_preview_resize)
        self.last_preview_width = 100
        self.last_preview_height = 100


    def _on_preview_resize(self, event):
        # Store the new size. Image update happens after generation.
        self.last_preview_width = event.width
        self.last_preview_height = event.height

    def _create_status_bar(self):
        """Creates the status bar at the bottom."""
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="2 5")
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _browse_model_file(self):
        """Opens file dialog to select a model (.pth) file."""
        filepath = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=(("PyTorch Model", "*.pth"), ("All files", "*.*"))
        )
        if filepath:
            self.model_path_var.set(filepath)
            self._update_status(f"Model selected: {os.path.basename(filepath)}")
            # Optionally trigger model loading here or wait until generation
            # self._load_model_async() # Example if loading immediately

    def _browse_output_dir(self):
        """Opens directory dialog to select an output folder."""
        dirpath = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir_var.get() or os.getcwd()
        )
        if dirpath:
            self.output_dir_var.set(dirpath)
            self._update_status(f"Output directory set: {dirpath}")

    def _validate_inputs(self) -> bool:
        """Checks if required inputs are present."""
        if not self.model_path_var.get():
            messagebox.showerror("Input Error", "Please select a model file.")
            return False
        if not os.path.exists(self.model_path_var.get()):
             messagebox.showerror("Input Error", f"Model file not found:\n{self.model_path_var.get()}")
             return False
        if not self.text_input.get("1.0", tk.END).strip():
            messagebox.showerror("Input Error", "Please enter text to generate.")
            return False
        if not self.output_dir_var.get():
             messagebox.showerror("Input Error", "Please select an output directory.")
             return False
        return True

    def _update_status(self, message: str):
        """Safely updates the status bar text from any thread."""
        self.status_var.set(message)

    def _update_preview_image(self, pil_image):
        """Safely updates the preview image from any thread."""
        try:
            img_w, img_h = pil_image.size
            lbl_w, lbl_h = self.last_preview_width, self.last_preview_height

            # Maintain aspect ratio, fit within label bounds
            if lbl_w < 10 or lbl_h < 10: return # Avoid division by zero if label size is tiny

            scale = min(lbl_w / img_w, lbl_h / img_h)
            new_w = int(img_w * scale * 0.95) # Add slight padding
            new_h = int(img_h * scale * 0.95)

            if new_w > 0 and new_h > 0:
                 resized_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                 self.preview_image_tk = ImageTk.PhotoImage(resized_image) # Keep reference!
                 self.preview_label.config(image=self.preview_image_tk)
            else:
                 logger.warning("Preview resize resulted in zero dimensions.")
                 self.preview_label.config(image='') # Clear image

        except Exception as e:
            logger.error(f"Failed to update preview image: {e}", exc_info=True)
            self._update_status(f"Error: Failed to display preview ({e}).")
            self.preview_label.config(image='') # Clear image


    def _start_generation(self):
        """Validates inputs and starts the generation process in a new thread."""
        if not self._validate_inputs():
            return

        # Prevent starting another thread if one is running
        if self.generator_thread and self.generator_thread.is_alive():
             messagebox.showwarning("Busy", "Generation is already in progress.")
             return

        self.generate_button.config(state=tk.DISABLED)
        self._update_status("Generating...")
        self.preview_label.config(image='') # Clear previous preview

        # Gather parameters
        params = {
            "model_path": self.model_path_var.get(),
            "text": self.text_input.get("1.0", tk.END).strip(),
            "style": self.style_var.get(),
            "stroke_width": self.stroke_width_var.get(),
            "random_variation_scale": self.random_variation_scale_var.get(),
            "seed_str": self.seed_var.get(), # Get seed as string
            "output_dir": self.output_dir_var.get(),
            "output_format": self.output_format_var.get(),
        }

        # Start generation in a background thread
        self.generator_thread = threading.Thread(target=self._run_generation, args=(params,), daemon=True)
        self.generator_thread.start()


    def _run_generation(self, params: dict):
        """The actual generation logic, run in a background thread."""
        try:
            start_time = time.time()

            # --- Load Model (if not already loaded or path changed) ---
            # Simple check: reload if path differs. Could add more sophisticated caching.
            if self.loaded_model is None or self.model_path_var.get() != getattr(self.loaded_model, '_loaded_path', None):
                self.loaded_model = None # Clear previous model first
                self.after(0, self._update_status, f"Loading model: {os.path.basename(params['model_path'])}...")
                logger.info(f"Loading model from {params['model_path']}...")
                checkpoint = torch.load(params['model_path'], map_location=self.device)
                if 'model_hyperparams' not in checkpoint:
                     raise ModelLoadError("Model hyperparameters not found in checkpoint.")
                model_hyperparams = checkpoint['model_hyperparams']
                self.loaded_model = HandwritingVAEGAN(**model_hyperparams).to(self.device)
                if 'model_state_dict' not in checkpoint:
                    raise ModelLoadError("Model state_dict not found in checkpoint.")
                self.loaded_model.load_state_dict(checkpoint['model_state_dict'])
                self.loaded_model.eval()
                setattr(self.loaded_model, '_loaded_path', params['model_path']) # Store path for comparison
                logger.info("Model loaded successfully.")
                self.after(0, self._update_status, "Model loaded. Generating...")

            # --- Initialize Generator ---
            seed_val = None
            if params['seed_str'].isdigit():
                seed_val = int(params['seed_str'])

            generator = HandwritingGenerator(
                model=self.loaded_model,
                device=self.device,
                style=params['style'],
                random_variation_scale=params['random_variation_scale'],
                stroke_width=params['stroke_width'],
                seed=seed_val
                # Extract feature indices if needed/configurable? Assume defaults for now.
            )

            # --- Generate SVG Content ---
            svg_content = generator.generate_handwriting(params['text'])

            # --- Save Output & Prepare Preview ---
            os.makedirs(params['output_dir'], exist_ok=True)
            base_filename = f"handwriting_{time.strftime('%Y%m%d_%H%M%S')}"
            output_filepath = os.path.join(params['output_dir'], f"{base_filename}.{params['output_format']}")

            preview_image_pil = None # Pillow Image object for preview

            # Save SVG first
            svg_filepath = os.path.join(params['output_dir'], f"{base_filename}.svg")
            try:
                 with open(svg_filepath, "w", encoding='utf-8') as f:
                      f.write(svg_content)
                 logger.info(f"SVG saved to {svg_filepath}")
                 if params['output_format'] == 'svg':
                     output_filepath = svg_filepath # Final output is SVG
            except Exception as e:
                 raise GenerationError(f"Failed to save SVG file: {e}") from e

            # Generate PNG/JPG if requested, and generate PNG for preview
            png_preview_bytes = None
            if params['output_format'] in ['png', 'jpg'] or True: # Always generate PNG for preview
                try:
                    logger.info("Rendering SVG to PNG for preview/output...")
                    # Render SVG content (string or bytes) to PNG bytes
                    png_preview_bytes = cairosvg.svg2png(
                        bytestring=svg_content.encode('utf-8'),
                        write_to=None, # In memory
                        dpi=150 # Preview DPI
                        # background_color='white' # Add background for preview?
                    )
                    if png_preview_bytes:
                         preview_image_pil = Image.open(io.BytesIO(png_preview_bytes))
                         logger.info("Rendered PNG for preview.")

                    # Save PNG/JPG file if requested format
                    if params['output_format'] == 'png':
                         with open(output_filepath, 'wb') as f: f.write(png_preview_bytes)
                         logger.info(f"PNG saved to {output_filepath}")
                    elif params['output_format'] == 'jpg':
                         if preview_image_pil:
                              # Convert to RGB and save as JPG
                              rgb_im = preview_image_pil.convert('RGB')
                              rgb_im.save(output_filepath, 'JPEG', quality=95)
                              logger.info(f"JPG saved to {output_filepath}")
                         else:
                              logger.error("Cannot save JPG, failed to render PNG first.")

                except ImportError:
                     self.after(0, self._update_status, "Error: CairoSVG needed for PNG/JPG output & preview.")
                     logger.error("CairoSVG required but not found during output/preview generation.")
                except Exception as e:
                     logger.error(f"Failed to convert SVG to raster: {e}", exc_info=True)
                     self.after(0, self._update_status, f"Error: Raster conversion failed ({e}).")


            # --- Update GUI (from main thread) ---
            if preview_image_pil:
                self.after(0, self._update_preview_image, preview_image_pil)

            end_time = time.time()
            elapsed = end_time - start_time
            self.after(0, self._update_status, f"Generation complete ({elapsed:.2f}s). Output saved to '{os.path.basename(output_filepath)}'.")

        except (ModelLoadError, GenerationError, ConversionError, FileNotFoundError) as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            # Show error in message box and status bar
            self.after(0, messagebox.showerror, "Generation Error", str(e))
            self.after(0, self._update_status, f"Error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during generation: {e}", exc_info=True)
            self.after(0, messagebox.showerror, "Unexpected Error", f"An unexpected error occurred:\n{e}")
            self.after(0, self._update_status, f"Unexpected Error: {e}")
        finally:
            # --- Re-enable button (from main thread) ---
            self.after(0, lambda: self.generate_button.config(state=tk.NORMAL))


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Apply a theme if available (e.g., 'clam', 'alt', 'default', 'classic')
        style = ttk.Style()
        available_themes = style.theme_names()
        # Prefer modern themes if available
        for theme in ['clam', 'alt', 'default']:
             if theme in available_themes:
                  style.theme_use(theme)
                  break

        # Create and run the app
        app = HandwritingApp()
        app.mainloop()
    except Exception as e:
        # Catch init errors (e.g., missing libraries before mainloop)
        logger.critical(f"Failed to launch GUI application: {e}", exc_info=True)
        # Fallback error display if messagebox failed earlier
        print(f"FATAL ERROR: {e}")