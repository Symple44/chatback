from typing import Dict, Optional
from langchain.prompts import PromptTemplate
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    def __init__(self):
        """Initialise le gestionnaire de prompts."""
        self.templates = {
            "chat": PromptTemplate(
                template=settings.CHAT_TEMPLATE,
                input_variables=["system", "query", "context", "context_summary"]
            ),
            "system": settings.SYSTEM_PROMPT
        }

    def format_prompt(
        self,
        template_name: str,
        variables: Dict[str, str],
        language: str = "fr"
    ) -> str:
        """
        Formate un prompt en utilisant le template spécifié.
        """
        try:
            if template_name not in self.templates:
                raise ValueError(f"Template inconnu: {template_name}")

            template = self.templates[template_name]
            
            # Adaptation du prompt système selon la langue
            if template_name == "chat":
                variables["system"] = self.templates["system"].format(language=language)
            
            # Formatage du prompt
            formatted_prompt = template.format(**variables)
            logger.debug(f"Prompt formaté pour {template_name}")
            
            return formatted_prompt
            
        except Exception as e:
            logger.error(f"Erreur lors du formatage du prompt {template_name}: {e}")
            return ""

    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """Récupère un template de prompt."""
        return self.templates.get(template_name)

    def add_template(self, name: str, template: str, variables: List[str]):
        """Ajoute un nouveau template."""
        try:
            self.templates[name] = PromptTemplate(
                template=template,
                input_variables=variables
            )
            logger.info(f"Nouveau template ajouté: {name}")
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du template {name}: {e}")