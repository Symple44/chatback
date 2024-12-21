# core/cache.py
import redis
import json
import zlib
from typing import Optional, Any, Dict
from datetime import datetime
import logging
from core.config import settings
from core.utils.logger import get_logger
from core.utils.metrics import metrics
import asyncio

logger = get_logger("cache")

class RedisCache:
    def __init__(self):
        """
        Initialise la connexion Redis avec les paramètres de configuration.
        """
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD,
                decode_responses=False,  # False pour stocker des données binaires (compression)
                socket_timeout=5.0,
                socket_keepalive=True,
                socket_connect_timeout=2.0,
                retry_on_timeout=True,
                health_check_interval=30
            )
            self.default_expiry = 7200  # 2 heures
            
            # Vérification initiale
            if self.redis_client.ping():
                logger.info("Connexion Redis initialisée avec succès")
                metrics.increment_counter("redis_init_success")
            else:
                metrics.increment_counter("redis_init_failed")
                raise ConnectionError("Échec du ping Redis")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de Redis: {e}")
            metrics.increment_counter("redis_init_error")
            raise

    async def get(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache et la décompresse.
        """
        with metrics.timer("redis_get"):
            try:
                data = self.redis_client.get(key)
                if data is not None:
                    decompressed_data = zlib.decompress(data).decode('utf-8')
                    result = json.loads(decompressed_data)
                    
                    # Ajout des métadonnées de cache
                    result['_cache_metadata'] = {
                        'cached_at': result.get('_cache_metadata', {}).get('cached_at'),
                        'retrieved_at': datetime.utcnow().isoformat(),
                        'cache_key': key
                    }
                    
                    metrics.increment_counter("cache_hits")
                    logger.debug(f"Cache hit pour la clé: {key}")
                    return result
                    
                metrics.increment_counter("cache_misses")
                logger.debug(f"Cache miss pour la clé: {key}")
                return None
                
            except zlib.error as e:
                logger.error(f"Erreur de décompression pour la clé {key}: {e}")
                metrics.increment_counter("cache_decompression_errors")
                return None
                
            except json.JSONDecodeError as e:
                logger.error(f"Erreur de décodage JSON pour la clé {key}: {e}")
                metrics.increment_counter("cache_json_decode_errors")
                return None
                
            except Exception as e:
                logger.error(f"Erreur lors de la récupération du cache ({key}): {e}")
                metrics.increment_counter("cache_errors")
                return None

    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None,
        compress_level: int = 6
    ) -> bool:
        """
        Définit une valeur dans le cache en compressant les données.
        """
        with metrics.timer("redis_set"):
            try:
                # Ajout des métadonnées de cache
                if isinstance(value, dict):
                    value['_cache_metadata'] = {
                        'cached_at': datetime.utcnow().isoformat(),
                        'cache_key': key
                    }
                
                # Sérialisation et compression
                serialized_data = json.dumps(value).encode('utf-8')
                compressed_data = zlib.compress(serialized_data, level=compress_level)
                
                # Stockage dans Redis
                success = self.redis_client.setex(
                    key,
                    expire or self.default_expiry,
                    compressed_data
                )
                
                if success:
                    metrics.increment_counter("cache_sets_success")
                    logger.debug(f"Données mises en cache avec succès pour la clé: {key}")
                    return True
                    
                metrics.increment_counter("cache_sets_failed")
                return False
                
            except (TypeError, ValueError) as e:
                logger.error(f"Erreur de sérialisation pour la clé {key}: {e}")
                metrics.increment_counter("cache_serialization_errors")
                return False
                
            except Exception as e:
                logger.error(f"Erreur lors de la mise en cache ({key}): {e}")
                metrics.increment_counter("cache_errors")
                return False

    async def check_connection(self) -> bool:
        """
        Vérifie la connexion à Redis.
        """
        try:
            with metrics.timer("redis_check_connection"):
                result = await asyncio.get_running_loop().run_in_executor(
                    None, 
                    self.redis_client.ping
                )
                if result:
                    metrics.increment_counter("redis_connection_check_success")
                else:
                    metrics.increment_counter("redis_connection_check_failed")
                return result
        except Exception as e:
            logger.error(f"Erreur de connexion Redis: {e}")
            metrics.increment_counter("redis_connection_check_error")
            return False

    async def delete(self, key: str) -> bool:
        """
        Supprime une valeur du cache.
        """
        try:
            with metrics.timer("redis_delete"):
                if self.redis_client.exists(key):
                    success = bool(self.redis_client.delete(key))
                    if success:
                        metrics.increment_counter("cache_deletes_success")
                        logger.debug(f"Clé supprimée du cache: {key}")
                    else:
                        metrics.increment_counter("cache_deletes_failed")
                    return success
                    
                logger.debug(f"Clé non trouvée dans le cache: {key}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du cache ({key}): {e}")
            metrics.increment_counter("cache_errors")
            return False

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques du cache Redis.
        """
        try:
            with metrics.timer("redis_stats"):
                info = self.redis_client.info()
                return {
                    "used_memory": info.get("used_memory_human", "N/A"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_keys": self.redis_client.dbsize(),
                    "uptime_seconds": info.get("uptime_in_seconds", 0),
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0),
                    "evicted_keys": info.get("evicted_keys", 0)
                }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats Redis: {e}")
            metrics.increment_counter("redis_stats_error")
            return {}

    async def clear_all(self) -> bool:
        """
        Supprime toutes les données dans Redis.
        """
        try:
            with metrics.timer("redis_clear_all"):
                self.redis_client.flushdb()
                logger.info("Cache Redis vidé avec succès")
                metrics.increment_counter("cache_clear_success")
                return True
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage du cache: {e}")
            metrics.increment_counter("cache_clear_failed")
            return False

    def __del__(self):
        """
        Nettoyage des ressources lors de la destruction de l'objet.
        """
        try:
            self.redis_client.close()
            logger.info("Connexion Redis fermée proprement")
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture de la connexion Redis: {e}")
