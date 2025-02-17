# core/streaming/stream_manager.py
from typing import AsyncGenerator, Any, Dict, List
from datetime import datetime
from core.utils.metrics import metrics
from core.utils.logger import get_logger
from core.config.config import settings
import json
import asyncio

logger = get_logger("stream_manager")

class StreamManager:
    """Gestionnaire de streaming avec optimisation des ressources."""
    
    def __init__(self):
        """Initialise le gestionnaire de streaming."""
        self.batch_size = settings.STREAM_BATCH_SIZE
        self.max_queue_size = settings.STREAM_MAX_QUEUE_SIZE
        self.timeout = settings.STREAM_TIMEOUT
        self.queue = asyncio.Queue(maxsize=self.max_queue_size)
        
    async def stream_response(
        self,
        query: str,
        model,
        context_docs: List[Dict] = None,
        **kwargs
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream une réponse avec gestion optimisée de la mémoire et du réseau.
        """
        try:
            token_batch = []
            start_time = datetime.utcnow()
            
            async for token in model.generate_streaming_response(
                query=query,
                context_docs=context_docs,
                **kwargs
            ):
                # Accumulation des tokens
                token_batch.append(token)
                
                # Envoi du batch quand il atteint la taille définie
                if len(token_batch) >= self.batch_size:
                    yield {
                        "event": "token",
                        "data": json.dumps({
                            "content": "".join(token_batch),
                            "type": "token_batch",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    }
                    token_batch = []
                    
                    # Gestion de la back-pressure
                    if self.queue.qsize() >= self.max_queue_size:
                        await asyncio.sleep(0.1)
            
            # Envoi des tokens restants
            if token_batch:
                yield {
                    "event": "token",
                    "data": json.dumps({
                        "content": "".join(token_batch),
                        "type": "token_batch",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                }
            
            # Envoi des métadonnées finales
            yield {
                "event": "complete",
                "data": json.dumps({
                    "status": "completed",
                    "metadata": {
                        "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                        "tokens_generated": len(token_batch),
                        "stream_config": {
                            "batch_size": self.batch_size,
                            "max_queue_size": self.max_queue_size
                        }
                    }
                })
            }
            
        except Exception as e:
            logger.error(f"Erreur streaming: {e}")
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
            }