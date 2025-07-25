from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QMessageBox, 
                           QFileDialog, QVBoxLayout, QWidget)
from PyQt5.QtCore import QSettings
import os

from src.ui.camera_tab import CameraTab
from src.ui.background_tab import BackgroundTab
from src.ui.difference_tab import DifferenceTab

from src.camera.camera_manager import CameraManager
from src.camera.image_reader import ImageReader
from src.analysis.image_analyzer import ImageAnalyzer
from src.visualizer.plot_manager import PlotManager
from src.data.exporter import DataExporter
from src.data.importer import DataImporter

class MainWindow(QMainWindow):
    """
    Main window of the Beam Profile Analyzer application
    """
    def __init__(self):
        super().__init__()
        
        # Initialization of managers and analyzers
        self.camera_manager = CameraManager()
        self.image_reader = ImageReader(self.camera_manager)
        self.image_analyzer = ImageAnalyzer()
        self.plot_manager = PlotManager()
        
        # Application settings
        self.settings = QSettings("BeamProfileAnalyzer", "BeamProfileAnalyzer")
        
        # Data buffers for different modes
        self.camera_data = {
            "current_frame": None,
            "background": None,
            "difference": None,
            "centroid": (0, 0),
            "rms": (0, 0)
        }
        
        self.file_data = {
            "current_frame": None,
            "background": None,
            "difference": None,
            "centroid": (0, 0),
            "rms": (0, 0),
            "filepath": None
        }
        
        # Operation mode (camera or file)
        self.current_mode = "camera"
        
        # UI setup
        self.setWindowTitle("Beam Profile Analyzer")
        self.resize(1200, 800)
        
        # Creating tabs
        self.tabs = QTabWidget()
        self.camera_tab = CameraTab(self)
        self.background_tab = BackgroundTab(self)
        self.difference_tab = DifferenceTab(self)
        
        # Adding tabs to the container
        self.tabs.addTab(self.camera_tab, "Main Capture")
        self.tabs.addTab(self.background_tab, "Background")
        self.tabs.addTab(self.difference_tab, "Difference")
        
        # Setting the central widget
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.tabs)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        # Connecting signals
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
    def on_tab_changed(self, index):
        """
        Handler for tab change
        
        Args:
            index: Index of the new tab
        """
        # Update only the active tab
        active_tab = self.tabs.widget(index)
        active_tab.update_tab()
        
        # Stop update timers in inactive tabs
        for i in range(self.tabs.count()):
            if i != index:
                tab = self.tabs.widget(i)
                if hasattr(tab, 'update_timer'):
                    tab.update_timer.stop()
        
        # Start the update timer only for the active tab
        if hasattr(active_tab, 'update_timer'):
            active_tab.update_timer.start(1000)  # Increase interval to 1 second
        
    def switch_mode(self, mode):
        """
        Switches the application operation mode
        
        Args:
            mode: Operation mode ("camera" or "file")
        """
        if mode not in ["camera", "file"]:
            return
            
        self.current_mode = mode
        
        # Update all tabs
        self.camera_tab.update_tab()
        self.background_tab.update_tab()
        self.difference_tab.update_tab()
        
    def connect_to_camera(self, camera_name):
        """
        Connects to the selected camera
        
        Args:
            camera_name: Camera name
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        # Disconnect from the current camera if any
        self.disconnect_camera()
        
        # Connect to the selected camera
        result = self.camera_manager.connect_to_camera(camera_name)
        
        if result:
            # Start preview
            self.image_reader.start_preview()
            
            # Switch to camera mode
            self.switch_mode("camera")
            
        return result
        
    def disconnect_camera(self):
        """
        Disconnects from the camera
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        # Stop preview
        self.image_reader.stop_preview()
        
        # Disconnect from camera
        return self.camera_manager.disconnect_camera()
        
    def start_background_collection(self, frames_count=40):
        """
        Starts collecting frames for background
        
        Args:
            frames_count: Number of frames to collect
            
        Returns:
            bool: True if start successful, False otherwise
        """
        return self.image_reader.start_background_collection(frames_count)
        
    def stop_background_collection(self):
        """
        Stops collecting frames for background
        
        Returns:
            bool: True if stop successful, False otherwise
        """
        result = self.image_reader.stop_background_collection()
        
        # Check if background was successfully collected
        has_background = self.image_reader.background is not None
        print("MainWindow: Background collected: {}".format(has_background))
        
        # Force update all tabs after background collection
        print("MainWindow: Forcing update of all tabs after background collection")
        self.camera_tab.update_tab()
        self.background_tab.update_tab()
        self.difference_tab.update_tab()
        
        # Switch to background tab to view result
        print("MainWindow: Switching to background tab")
        self.tabs.setCurrentWidget(self.background_tab)
        
        return result
        
    def get_current_data(self):
        """
        Returns current data depending on the operation mode
        
        Returns:
            dict: Dictionary with current data
        """
        # Update data if in camera mode
        if self.current_mode == "camera":
            # Get current frame
            self.camera_data["current_frame"] = self.image_reader.get_current_frame()
            self.camera_data["background"] = self.image_reader.get_background()
            self.camera_data["difference"] = self.image_reader.get_difference()
            
            # If there is a current frame, calculate centroid and RMS
            if self.camera_data["current_frame"] is not None:
                camera_info = self.camera_manager.get_camera_info()
                if camera_info:
                    pixel_size_x = camera_info["pixel_size_x"]
                    pixel_size_y = camera_info["pixel_size_y"]
                    
                    self.camera_data["centroid"] = self.image_analyzer.calculate_centroid(
                        self.camera_data["current_frame"],
                        pixel_size_x,
                        pixel_size_y
                    )
                    
                    self.camera_data["rms"] = self.image_analyzer.calculate_rms(
                        self.camera_data["current_frame"],
                        pixel_size_x,
                        pixel_size_y
                    )
            
            return self.camera_data
        else:
            # In file mode, data is static, so no need to recalculate centroid and RMS
            return self.file_data
            
    def get_camera_info(self):
        """
        Returns camera information
        
        Returns:
            dict: Camera information or None if camera is not selected
        """
        if self.current_mode == "camera":
            if self.camera_manager.is_connected:
                return self.camera_manager.get_camera_info()
            else:
                return None
        else:
            # In file mode, get information from file data
            if self.file_data["current_frame"] is not None:
                return {
                    "resolution": self.file_data.get("resolution", (0, 0)),
                    "pixel_size_x": self.file_data.get("pixel_size_x", 0),
                    "pixel_size_y": self.file_data.get("pixel_size_y", 0),
                    "camera_name": self.file_data.get("camera_name", "Unknown camera")
                }
            return None
            
    def export_data(self, filepath, export_format="folder"):
        """
        Exports data to the specified format
        
        Args:
            filepath: Path to save file/directory
            export_format: Export format ("folder" for CSV+JSON, "png")
            
        Returns:
            str: Path to the created data folder or None if export failed
        """
        # Get current data
        data = self.get_current_data()
        camera_info = self.get_camera_info()
        
        # If there is no data or camera information
        if data["current_frame"] is None or camera_info is None:
            QMessageBox.warning(self, "Warning", "No data to export")
            return None
            
        # Form export dictionary
        export_data = {
            "shot": data["current_frame"],
            "background": data["background"],
            "difference": data["difference"],
            "raw_shot": self.image_reader.get_raw_current_frame() if self.current_mode == "camera" 
                       else data.get("raw_shot"),
            "raw_background": self.image_reader.get_raw_background() if self.current_mode == "camera"
                            else data.get("raw_background"),
            "resolution": camera_info["resolution"],
            "pixel_size_x": camera_info["pixel_size_x"],
            "pixel_size_y": camera_info["pixel_size_y"],
            "camera_name": camera_info.get("camera_name", "Unknown camera")
        }
        
        # Export data
        result = DataExporter.export_data(filepath, export_data, export_format, self.plot_manager)
        
        if result:
            QMessageBox.information(self, "Information", "Data successfully exported to {}".format(result))
        else:
            QMessageBox.warning(self, "Warning", "Error exporting data")
            
        return result
        
    def import_data(self, filepath):
        """
        Imports data from a file
        
        Args:
            filepath: Path to file
            
        Returns:
            bool: True if import successful, False otherwise
        """
        try:
            # Stop update timers in all tabs
            if hasattr(self, 'camera_tab') and self.camera_tab:
                self.camera_tab.update_timer.stop()
            if hasattr(self, 'background_tab') and self.background_tab:
                self.background_tab.update_timer.stop()
            if hasattr(self, 'difference_tab') and self.difference_tab:
                self.difference_tab.update_timer.stop()
            
            # Import data
            data = DataImporter.import_data(filepath)
            
            if data is None:
                QMessageBox.critical(self, "Error", "Failed to import file: {}".format(filepath))
                return False
            
            # Switch to file mode
            self.current_mode = "file"
            
            # Initialize file_data with basic structure to avoid KeyError
            self.file_data = {
                "shot": None,
                "background": None,
                "difference": None,
                "current_frame": None,
                "raw_shot": None,
                "raw_background": None,
                "resolution": (0, 0),
                "pixel_size_x": 1.0,
                "pixel_size_y": 1.0,
                "camera_name": "Unknown camera",
                "centroid": (0, 0),
                "rms": (0, 0),
                "filepath": filepath
            }
            
            # Update data from imported dictionary
            self.file_data.update(data)
            
            # Copy shot to current_frame, since camera tab uses current_frame key
            if "shot" in data and data["shot"] is not None:
                self.file_data["current_frame"] = data["shot"]
            
            # Get pixel size from data
            pixel_size_x = self.file_data.get("pixel_size_x", 1)
            pixel_size_y = self.file_data.get("pixel_size_y", 1)
            
            try:
                # Calculate centroid and RMS for main image
                if self.file_data["current_frame"] is not None:
                    self.file_data["centroid"] = self.image_analyzer.calculate_centroid(
                        self.file_data["current_frame"],
                        pixel_size_x,
                        pixel_size_y
                    )
                    
                    self.file_data["rms"] = self.image_analyzer.calculate_rms(
                        self.file_data["current_frame"],
                        pixel_size_x,
                        pixel_size_y
                    )
                
                # Calculate centroid and RMS for background image
                if self.file_data["background"] is not None:
                    self.file_data["background_centroid"] = self.image_analyzer.calculate_centroid(
                        self.file_data["background"],
                        pixel_size_x,
                        pixel_size_y
                    )
                    
                    self.file_data["background_rms"] = self.image_analyzer.calculate_rms(
                        self.file_data["background"],
                        pixel_size_x,
                        pixel_size_y
                    )
                
                # Calculate centroid and RMS for image difference
                if self.file_data["difference"] is not None:
                    self.file_data["difference_centroid"] = self.image_analyzer.calculate_centroid(
                        self.file_data["difference"],
                        pixel_size_x,
                        pixel_size_y
                    )
                    
                    self.file_data["difference_rms"] = self.image_analyzer.calculate_rms(
                        self.file_data["difference"],
                        pixel_size_x,
                        pixel_size_y
                    )
            except Exception as e:
                # Error in calculations should not interrupt file load,
                # but should be logged and shown to user
                print("Error in calculation parameters: {}".format(e))
                QMessageBox.warning(self, "Warning", 
                                   "File loaded, but unable to calculate parameters: {}".format(str(e)))
            
            # Update all tabs
            self.camera_tab.update_tab()
            self.background_tab.update_tab()
            self.difference_tab.update_tab()
            
            # Start update timers
            if hasattr(self, 'camera_tab') and self.camera_tab:
                self.camera_tab.update_timer.start(500)
            if hasattr(self, 'background_tab') and self.background_tab:
                self.background_tab.update_timer.start(500)
            if hasattr(self, 'difference_tab') and self.difference_tab:
                self.difference_tab.update_timer.start(500)
            
            return True
        
        except Exception as e:
            # In case of error, also restore timers
            if hasattr(self, 'camera_tab') and self.camera_tab:
                self.camera_tab.update_timer.start(500)
            if hasattr(self, 'background_tab') and self.background_tab:
                self.background_tab.update_timer.start(500)
            if hasattr(self, 'difference_tab') and self.difference_tab:
                self.difference_tab.update_timer.start(500)
            
            # Show detailed error message
            error_message = "Error importing file {}:\n{}".format(filepath, str(e))
            print(error_message)
            QMessageBox.critical(self, "Import Error", error_message)
            
            return False
        
    def open_file_dialog(self):
        """
        Открывает диалог выбора формата данных (MAT или папка с данными)
        
        Returns:
            bool: True если данные успешно импортированы, False иначе
        """
        # Фильтры для выбора типа файлов
        filters = [
            "MAT файлы (*.mat)",
            "Папки с данными (metadata.json)"
        ]
        filter_string = ";;".join(filters)
        
        # Открываем диалог выбора файла
        filepath, selected_filter = QFileDialog.getOpenFileName(
            self,
            "Импорт данных",
            "",
            filter_string,
            filters[0]  # По умолчанию MAT-файлы
        )
        
        if not filepath:
            return False
            
        # Определяем выбранный формат
        if selected_filter == filters[0]:  # MAT файлы
            # Импортируем MAT-файл напрямую
            return self.import_data(filepath)
        else:  # Папки с данными
            # Проверяем, выбран ли файл metadata.json
            if os.path.basename(filepath) == "metadata.json":
                # Импортируем данные из папки
                return self.import_data(filepath)
            else:
                QMessageBox.warning(
                    self,
                    "Неверный выбор",
                    "Для импорта папки с данными необходимо выбрать файл metadata.json из этой папки."
                )
                return False
        
    def save_file_dialog(self, default_format="folder"):
        """
        Opens dialog for exporting data to a folder
        
        Args:
            default_format: Default export format ("folder", "png")
            
        Returns:
            str: Path to the created data folder or None if export cancelled or failed
        """
        # Formats and filters
        formats = {
            "folder": "Data folder (CSV+JSON)",
            "png": "PNG images folder"
        }
        
        # Filter string
        filter_str = ";;".join(formats.values())
        
        # Select default format
        default_filter = formats.get(default_format, formats["folder"])
        
        # Open save dialog - using getSaveFileName but will create a folder
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save data to folder",
            "",
            filter_str,
            default_filter
        )
        
        if not filepath:
            return None
            
        # Determine selected format
        export_format = None
        for fmt, filter_name in formats.items():
            if filter_name == selected_filter:
                export_format = fmt
                break
                
        if export_format is None:
            export_format = default_format
            
        # Export data - will create a folder at the specified path
        return self.export_data(filepath, export_format)
        
    def closeEvent(self, event):
        """
        Handler for window close
        
        Args:
            event: Event object
        """
        # Disconnect from camera
        self.disconnect_camera()
        
        # Save settings
        self.settings.sync()
        
        event.accept()
