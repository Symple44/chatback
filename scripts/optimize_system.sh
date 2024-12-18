#!/bin/bash
# scripts/optimize_system.sh

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Vérification si exécuté en tant que root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_warning "Script exécuté sans privilèges root - certaines optimisations seront limitées"
        return 1
    fi
    return 0
}

# Optimisation de la mémoire système
optimize_memory() {
    log_info "Optimisation de la mémoire système..."
    
    # Tente de définir swappiness si root
    if check_root; then
        echo 60 > /proc/sys/vm/swappiness 2>/dev/null && \
            log_info "Swappiness définie à 60" || \
            log_warning "Impossible de modifier swappiness"
            
        # Nettoyage des caches système
        sync
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null && \
            log_info "Caches système nettoyés" || \
            log_warning "Impossible de nettoyer les caches système"
    fi

    # Optimisations ne nécessitant pas root
    # Limite de mémoire pour le processus Python actuel
    ulimit -v 22000000 2>/dev/null && \
        log_info "Limite mémoire virtuelle définie à ~22GB" || \
        log_warning "Impossible de définir la limite mémoire virtuelle"
}

# Optimisation CPU
optimize_cpu() {
    log_info "Optimisation CPU..."
    
    # Limite le nombre de threads Python
    export NUMEXPR_MAX_THREADS=8
    export MKL_NUM_THREADS=8
    export OPENBLAS_NUM_THREADS=8
    export OMP_NUM_THREADS=8
    
    log_info "Nombre de threads limité à 8 pour les bibliothèques numériques"
}

# Optimisation des fichiers système
optimize_files() {
    log_info "Optimisation des fichiers système..."
    
    # Création des répertoires nécessaires
    mkdir -p logs model_cache documents
    
    # Nettoyage des fichiers temporaires Python
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    
    log_info "Nettoyage des fichiers temporaires effectué"
}

# Fonction principale
main() {
    log_info "Démarrage de l'optimisation système..."
    
    optimize_memory
    optimize_cpu
    optimize_files
    
    log_info "Optimisation système terminée"
}

# Exécution
main