# core/config/hardware/detector.py
from typing import Dict, Any
import os
import platform
import psutil

# Vérification de disponibilité CUDA
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

class HardwareDetector:
    """Détecteur de matériel pour la configuration automatique."""

    @staticmethod
    def detect_cpu() -> Dict[str, Any]:
        """Détecte les caractéristiques du CPU."""
        cpu_info = {
            "model": os.environ.get("CPU_MODEL", "Indéterminé"),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "architecture": platform.machine(),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        }
        
        # Tentative de détection du modèle si non spécifié
        if cpu_info["model"] == "Indéterminé":
            try:
                if platform.system() == "Linux":
                    with open("/proc/cpuinfo", "r") as f:
                        for line in f:
                            if "model name" in line:
                                cpu_info["model"] = line.split(":")[1].strip()
                                break
                elif platform.system() == "Darwin":  # macOS
                    import subprocess
                    cpu_info["model"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).strip().decode()
                elif platform.system() == "Windows":
                    import subprocess
                    cpu_info["model"] = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().split("\n")[1].strip()
            except Exception:
                pass
        
        return cpu_info

    @staticmethod
    def detect_gpu() -> Dict[str, Any]:
        """Détecte les caractéristiques du GPU."""
        gpu_info = {
            "available": CUDA_AVAILABLE,
            "name": os.environ.get("GPU_NAME", "Indéterminé"),
            "vram_gb": 0,
            "compute_capability": "0.0"
        }

        if CUDA_AVAILABLE:
            try:
                # Obtenir les infos du premier GPU (index 0)
                device = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(device)
                
                gpu_info.update({
                    "name": properties.name,
                    "vram_gb": round(properties.total_memory / (1024**3), 2),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "multi_processor_count": properties.multi_processor_count,
                    "cuda_version": torch.version.cuda
                })
            except Exception:
                pass
                
        return gpu_info

    @staticmethod
    def detect_hardware() -> Dict[str, Any]:
        """Détecte l'ensemble du matériel disponible."""
        cpu_info = HardwareDetector.detect_cpu()
        gpu_info = HardwareDetector.detect_gpu()
        
        # Création de l'objet hardware complet
        hardware = {
            "cpu": cpu_info,
            "gpu": gpu_info,
            "system": {
                "platform": platform.system(),
                "release": platform.release(),
                "version": platform.version()
            }
        }
        
        return hardware