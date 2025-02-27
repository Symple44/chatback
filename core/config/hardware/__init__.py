# core/config/hardware/__init__.py
from .detector import HardwareDetector
from .profiles import HardwareProfiles
from .cuda import CUDAConfig

class HardwareConfig:
    """Configuration unifiée pour le matériel."""
    
    def __init__(self):
        # Détection du matériel
        self.detector = HardwareDetector()
        
        # Obtention du profil matériel
        hardware_info = self.detector.detect_hardware()
        self.profile = HardwareProfiles.get_profile(hardware_info)
        
        # Configuration CUDA
        self.cuda = CUDAConfig.from_profile(self.profile)
        
        # Optimisation des threads
        self.thread_config = {
            "MKL_NUM_THREADS": self.profile["cpu"]["thread_config"]["workers"],
            "NUMEXPR_NUM_THREADS": self.profile["cpu"]["thread_config"]["workers"],
            "OMP_NUM_THREADS": self.profile["cpu"]["thread_config"]["inference_threads"],
            "OPENBLAS_NUM_THREADS": self.profile["cpu"]["thread_config"]["inference_threads"]
        }

__all__ = ['HardwareConfig']