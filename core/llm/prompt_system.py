# core/llm/prompt_system.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import re
from core.config import settings
from core.utils.logger import get_logger

logger = get_logger("prompt_system")

class Message(BaseModel):
    role: str = Field(..., description="Role du message (system, user, assistant)")
    content: str = Field(..., description="Contenu du message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)

class PromptSystem:
    """Gestionnaire de prompts pour le modèle."""
    
    def __init__(self):
        """Initialise le système de prompts."""
        # Template principal pour les conversations
        self.chat_template = settings.CHAT_TEMPLATE
        self.system_template = settings.SYSTEM_PROMPT

        # Templates spécifiques pour différents types de réponses
        self.templates = {
            "procedure": """
Instructions étape par étape :
{steps}

Points importants :
{notes}
""",
            "error": """
⚠️ Attention : {error_message}
Solutions possibles :
{solutions}
""",
            "context": """
Basé sur les documents suivants :
{context}

Éléments clés identifiés :
{key_points}
"""
        }
        
        # Formattage des rôles
        self.role_formats = {
            "system": "Instructions système :\n{message}",
            "user": "Question utilisateur :\n{message}",
            "assistant": "Réponse assistant :\n{message}"
        }
    
    def build_chain_of_thought_prompt(
        self,
        messages: List[Dict],
        context: Optional[str] = None,
        query: Optional[str] = None,
        lang: str = "fr"
    ) -> str:
        try:
            # Préparation du contexte
            context_summary = self._extract_key_points(context) if context else []
            
            # Construction du prompt de raisonnement structuré
            prompt = f"""Consignes de réponse :
    1. Analyse du contexte
    - Identifiez précisément les informations pertinentes pour répondre à la question
    - Repérez les éléments clés en lien direct avec la demande

    2. Décomposition logique
    - Décomposez la question en sous-parties
    - Associez chaque élément aux informations contextuelles
    - Évaluez la capacité du contexte à répondre complètement

    3. Formulation de la réponse
    - Construisez une réponse structurée et concise
    - Utilisez STRICTEMENT les informations du contexte
    - Répondez de manière claire et directe

    Contexte documentaire :
    {context or "Aucun contexte disponible"}

    Points clés du contexte :
    {" • ".join(context_summary) if context_summary else "Aucun point clé identifié"}

    Question à traiter :
    {query}

    Réponse :
    """
            
            return prompt

        except Exception as e:
            logger.error(f"Erreur génération prompt chain-of-thought: {e}")
            return f"Question: {query}\nRéponse:"

    def build_chat_prompt(
        self,
        messages: List[Dict],
        context: Optional[str] = None,
        query: Optional[str] = None,
        lang: str = "fr"
    ) -> str:
        """
        Construit le prompt complet pour une conversation.
        
        Args:
            messages: Historique des messages
            context: Contexte documentaire
            query: Question actuelle
            lang: Code de langue
            
        Returns:
            str: Prompt formaté
        """
        try:
            # Préparation du contexte système
            system = self.system_template.format(app_name=settings.APP_NAME)

            # Extraction des points clés du contexte
            if context:
                key_points = self._extract_key_points(context)
                context_summary = "• " + "\n• ".join(key_points)
            else:
                context_summary = "Pas de contexte documentaire disponible."

            # Construction de l'historique
            history = []
            if messages:
                for msg in messages[-5:]:  # Limite aux 5 derniers messages
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    history.append(
                        self.role_formats[role].format(message=content)
                    )

            # Construction du prompt final
            prompt = self.chat_template.format(
                system=system,
                query=query or "",
                context=context or "",
                context_summary=context_summary,
                history="\n\n".join(history),
                timestamp=datetime.utcnow().isoformat(),
                language=lang,
                response=""
            )

            return prompt

        except Exception as e:
            logger.error(f"Erreur construction prompt: {e}")
            # Fallback sur un prompt minimal
            return f"<|system|>\n{settings.SYSTEM_PROMPT}\n</|system|>\n\n<|user|>\n{query}\n</|user|>\n\n<|assistant|>\n"

    def _extract_key_points(self, context: str, max_points: int = 3) -> List[str]:
        """Extrait les points clés du contexte."""
        points = []
        
        # Patterns pour identifier les informations importantes
        patterns = [
            (r"Pour (créer|modifier|supprimer).*?[.]\n", 0.9),  # Instructions directes
            (r"^\d+\.\s+.*?[.]\n", 0.8),                        # Étapes numérotées
            (r"(?:Attention|Important)\s*:.*?[.]\n", 0.7),      # Avertissements
            (r"^[-•]\s+.*?[.]\n", 0.6),                        # Points listés
        ]

        # Extraction des correspondances
        for pattern, score in patterns:
            matches = re.finditer(pattern, context, re.MULTILINE)
            for match in matches:
                text = match.group(0).strip()
                if text and len(text) > 20:  # Filtre les segments trop courts
                    points.append((text, score))

        # Si pas assez de points, ajoute les premières phrases
        if len(points) < max_points:
            sentences = re.split(r'[.!?]\s+', context)
            for sentence in sentences:
                if len(sentence.strip()) > 20:
                    points.append((sentence.strip() + ".", 0.5))
                if len(points) >= max_points:
                    break

        # Tri par score et limite au nombre maximum
        points.sort(key=lambda x: x[1], reverse=True)
        return [point[0] for point in points[:max_points]]

    def format_message(self, message: Message) -> str:
        """Formate un message selon son rôle."""
        template = self.role_formats.get(message.role, "{message}")
        return template.format(message=message.content)

    def create_procedural_response(
        self,
        steps: List[str],
        notes: Optional[List[str]] = None
    ) -> str:
        """
        Crée une réponse procédurale structurée.
        
        Args:
            steps: Liste des étapes
            notes: Notes ou avertissements optionnels
            
        Returns:
            str: Réponse formatée
        """
        formatted_steps = "\n".join(
            f"{i+1}. {step}" for i, step in enumerate(steps)
        )
        
        formatted_notes = ""
        if notes:
            formatted_notes = "\n".join(f"• {note}" for note in notes)

        return self.templates["procedure"].format(
            steps=formatted_steps,
            notes=formatted_notes
        )

    def create_error_response(
        self,
        error_message: str,
        solutions: List[str]
    ) -> str:
        """
        Crée une réponse pour les situations d'erreur.
        
        Args:
            error_message: Message d'erreur
            solutions: Liste des solutions possibles
            
        Returns:
            str: Réponse formatée
        """
        formatted_solutions = "\n".join(
            f"{i+1}. {solution}" for i, solution in enumerate(solutions)
        )

        return self.templates["error"].format(
            error_message=error_message,
            solutions=formatted_solutions
        )

    def get_system_message(self, key: str, **kwargs) -> str:
        """Récupère un message système prédéfini."""
        messages = {
            "welcome": "Bienvenue ! Comment puis-je vous aider ?",
            "error": "Je suis désolé, une erreur est survenue.",
            "clarification": "Pouvez-vous préciser votre demande ?",
            "no_context": "Je n'ai pas trouvé de contexte pertinent.",
            "processing": "Je traite votre demande..."
        }
        
        message = messages.get(key, messages["error"])
        return message.format(**kwargs) if kwargs else message

    def truncate_messages(
        self,
        messages: List[Message],
        max_messages: Optional[int] = None
    ) -> List[Message]:
        """Tronque la liste des messages à un maximum."""
        if not max_messages:
            max_messages = settings.MAX_HISTORY_MESSAGES
            
        # Garde toujours le message système
        system_msg = next((m for m in messages if m.role == "system"), None)
        history = [m for m in messages if m.role != "system"]
        
        truncated = history[-max_messages:]
        if system_msg:
            truncated.insert(0, system_msg)
            
        return truncated
