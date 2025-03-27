from openai import OpenAI
from flask import current_app
from typing import List, Dict, Any, Optional

class AIService:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=current_app.config['OPENROUTER_API_KEY']
        )

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        image_url: Optional[str] = None
    ) -> str:
        """Create a chat completion using OpenRouter API with Gemini model"""
        try:
            # Prepare the messages format
            formatted_messages = []
            for msg in messages:
                if image_url and msg['role'] == 'user':
                    formatted_messages.append({
                        'role': msg['role'],
                        'content': [
                            {
                                'type': 'text',
                                'text': msg['content']
                            },
                            {
                                'type': 'image_url',
                                'image_url': {'url': image_url}
                            }
                        ]
                    })
                else:
                    formatted_messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })

            # Create chat completion
            completion = self.client.chat.completions.create(
                model=current_app.config['OPENROUTER_MODEL'],
                messages=formatted_messages,
                extra_headers={
                    'HTTP-Referer': current_app.config['OPENROUTER_SITE_URL'],
                    'X-Title': current_app.config['OPENROUTER_SITE_NAME']
                },
                extra_body={}
            )

            return completion.choices[0].message.content

        except Exception as e:
            current_app.logger.error(f"Error in chat completion: {str(e)}")
            return "I apologize, but I encountered an error processing your request. Please try again later."