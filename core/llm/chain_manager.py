from typing import Dict, List, Optional, Any
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.memory import BaseMemory
from langchain_core.messages import messages_from_dict, messages_to_dict
from pydantic import BaseModel
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class ChatEntry(BaseModel):
    input: str
    output: str

class ChatHistory(BaseModel):
    entries: List[ChatEntry] = []

class CustomMemory(BaseMemory):
    """Classe personnalisée pour la gestion de la mémoire."""
    
    def __init__(self):
        super().__init__()
        self._history = ChatHistory()
        self._memory_key = "chat_history"

    @property
    def memory_variables(self) -> List[str]:
        """Retourne la liste des variables de mémoire."""
        return [self._memory_key]
        
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Charge les variables de la mémoire."""
        return {
            self._memory_key: [
                {"input": entry.input, "output": entry.output}
                for entry in self._history.entries
            ]
        }
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Sauvegarde le contexte dans la mémoire."""
        if "question" in inputs and "answer" in outputs:
            self._history.entries.append(
                ChatEntry(
                    input=inputs["question"],
                    output=outputs["answer"]
                )
            )
    
    def clear(self) -> None:
        """Efface la mémoire."""
        self._history.entries.clear()

class LangchainManager:
    def __init__(self, llm, vectorstore=None):
        """Initialise le gestionnaire de chaînes Langchain."""
        try:
            self.llm = llm
            self.vectorstore = vectorstore
            
            # Utilisation de la mémoire personnalisée
            self.memory = CustomMemory()

            # Configuration du template de prompt
            self.prompt = ChatPromptTemplate.from_template(settings.CHAT_TEMPLATE)
            
            # La chaîne sera configurée quand le vectorstore sera disponible
            self.chain = None

            self.prompt_system = PromptSystem()
            
            logger.info("LangchainManager initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de LangchainManager: {e}")
            raise

    def setup_chain(self):
        """Configure la chaîne une fois que le vectorstore est disponible."""
        if self.vectorstore:
            self.chain = (
                {
                    "context": self.vectorstore.as_retriever(),
                    "query": RunnablePassthrough(),
                    "system": lambda _: settings.SYSTEM_PROMPT,  # Ajout de system
                    "context_summary": lambda x: "Résumé du contexte"  # Ajout du résumé
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )

    def set_vectorstore(self, vectorstore):
        """Configure le vectorstore et initialise la chaîne."""
        self.vectorstore = vectorstore
        self.setup_chain()

    async def process_query(
        self,
        query: str,
        context_docs: List[Dict],
        metadata_filter: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Traite une requête utilisateur avec la chaîne Langchain.
        """
        try:
            if not self.chain:
                raise ValueError("Chain not configured. Please set up vectorstore first.")
                
            prompt = self.prompt_system.build_chat_prompt(
                messages=[{"role": "user", "content": query}],
                context=self._prepare_context(context_docs),
                query=query
            )
            # Préparation du contexte
            context = self._prepare_context(context_docs)
            
            # Ajout des variables manquantes dans le prompt
            response = self.chain.invoke({
                "query": query,  # Changé de question à query
                "context": context,
                "system": settings.SYSTEM_PROMPT,
                "context_summary": "Résumé des documents pertinents"
            })
            
            # Sauvegarde de l'interaction dans la mémoire
            self.memory.save_context(
                {"query": query},  # Changé de question à query
                {"answer": response}
            )
            
            return {
                "response": response,
                "source_documents": context_docs,
                "confidence_score": self._calculate_confidence(context_docs)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la requête: {e}")
            raise

    def _prepare_context(self, documents: List[Dict]) -> str:
        """Prépare le contexte pour la chaîne."""
        context_parts = []
        for doc in documents:
            if "processed_sections" in doc:
                # Utilise les sections prétraitées
                sections = sorted(
                    doc["processed_sections"],
                    key=lambda x: x.get("importance_score", 0),
                    reverse=True
                )
                content = "\n\n".join(s["content"] for s in sections[:3])
            else:
                content = doc.get("content", "").strip()
                
            if content:
                title = doc.get("title", "Document")
                context_parts.append(f"[Source: {title}]\n{content}")
        
        return "\n\n---\n\n".join(context_parts) if context_parts else ""

    def _calculate_confidence(self, documents: List[Dict]) -> float:
        """Calcule le score de confiance."""
        if not documents:
            return 0.0
        scores = [doc.get('score', 0.0) for doc in documents]
        return sum(scores) / len(scores) if scores else 0.0

    def clear_memory(self):
        """Efface la mémoire de conversation."""
        try:
            self.memory.clear()
            logger.info("Mémoire de conversation effacée")
        except Exception as e:
            logger.error(f"Erreur lors de l'effacement de la mémoire: {e}")
