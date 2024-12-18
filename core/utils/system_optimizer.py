# core/utils/system_optimizer.py
import os
import subprocess
import logging
from pathlib import Path
from core.utils.logger import get_logger

logger = get_logger("system_optimizer")

class SystemOptimizer:
    def __init__(self):
        self.script_path = Path(__file__).parent.parent.parent / "scripts" / "optimize_system.sh"
        self.is_root = os.geteuid() == 0 if hasattr(os, 'geteuid') else False

    async def optimize(self) -> bool:
        """Exécute le script d'optimisation système."""
        try:
            if not self.script_path.exists():
                logger.error(f"Script d'optimisation non trouvé: {self.script_path}")
                return False

            # Configuration des variables d'environnement
            env = os.environ.copy()
            env.update({
                'NUMEXPR_MAX_THREADS': '8',
                'MKL_NUM_THREADS': '8',
                'OPENBLAS_NUM_THREADS': '8',
                'OMP_NUM_THREADS': '8'
            })

            # Exécution du script
            result = subprocess.run(
                ['/bin/bash', str(self.script_path)],
                env=env,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info("Optimisation système réussie")
                if result.stdout:
                    logger.debug(f"Sortie script: {result.stdout}")
                return True
            else:
                logger.warning(f"Script d'optimisation terminé avec code {result.returncode}")
                if result.stderr:
                    logger.error(f"Erreurs: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation système: {e}")
            return False

    def setup_environment(self):
        """Configure l'environnement d'exécution."""
        try:
            # Limite mémoire virtuelle si possible
            import resource
            resource.setrlimit(
                resource.RLIMIT_AS,
                (22 * 1024 * 1024 * 1024, -1)  # 22GB
            )
            logger.info("Limite mémoire virtuelle configurée")
        except Exception as e:
            logger.warning(f"Impossible de configurer la limite mémoire: {e}")

        # Configuration des threads
        os.environ['NUMEXPR_MAX_THREADS'] = '8'
        os.environ['MKL_NUM_THREADS'] = '8'
        os.environ['OPENBLAS_NUM_THREADS'] = '8'
        os.environ['OMP_NUM_THREADS'] = '8'
        logger.info("Variables d'environnement des threads configurées")