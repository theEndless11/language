from ctransformers import AutoModelForCausalLM
import requests
import time
import re
from typing import List, Dict, Optional

class SmartChatbot:
    def __init__(self, model_path="./mistral.Q4_K_M.gguf"):
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="mistral",
            threads=4,
            context_length=2048
        )
        
        # SearXNG instance
        self.searxng_url = "http://localhost:8888"
        
        self.conversation = []
        self.max_history = 6  # Keep last 6 exchanges
        
        print("Chatbot ready!")

    def should_search(self, user_input: str, conversation_context: str) -> bool:
        """Let the model naturally decide if web search is needed"""
        
        # Build a decision prompt that's more explicit
        decision_prompt = f"""Task: Decide if web search is needed for this question.

SEARCH (YES) for:
- Current time/date: "what day is today", "what time is it", "current date"
- Upcoming events: "upcoming movies", "new releases", "future concerts"
- Recent news: "latest news", "what happened", "recent events"
- Live data: "weather", "stock price", "scores"
- Current info about changing things: "who is the CEO of", "latest version of"

DON'T SEARCH (NO) for:
- Greetings: "hey", "hello", "hi there"
- About me: "what's your name", "who are you", "how are you"
- General knowledge: "what is gravity", "history of", "explain"
- Conversation: "thanks", "ok", "you're great"
- Opinions: "what do you think", "do you like"

Question: "{user_input}"

The user is asking about: (current/changing information that needs web search) OR (general conversation/knowledge I can handle)

Answer: YES or NO

Answer:"""
        
        try:
            tokens = self.model.tokenize(decision_prompt)
            response_tokens = []
            
            # Generate with temperature=0 for more consistent decisions
            for i, token in enumerate(self.model.generate(tokens, temperature=0.1, top_p=0.9)):
                response_tokens.append(token)
                current_text = self.model.detokenize(response_tokens).strip()
                
                # Stop as soon as we have YES or NO
                if current_text.upper().endswith("YES") or current_text.upper().endswith("NO"):
                    break
                    
                # Safety limit
                if i > 20:
                    break
            
            response = self.model.detokenize(response_tokens).strip().upper()
            
            # Look for YES/NO in the response
            if "YES" in response:
                return True
            elif "NO" in response:
                return False
            else:
                # If we can't get a clear decision, be conservative
                # For time/date questions, default to search
                time_indicators = ["day", "date", "time", "today", "now"]
                if any(word in user_input.lower() for word in time_indicators):
                    return True
                # For "upcoming" or "latest", default to search  
                current_indicators = ["upcoming", "latest", "recent", "new", "current"]
                if any(word in user_input.lower() for word in current_indicators):
                    return True
                return False
                
        except Exception as e:
            print(f"Search decision error: {e}")
            # Smart fallback based on key terms
            user_lower = user_input.lower()
            search_terms = ["upcoming", "latest", "today", "current", "recent", "what day", "what time"]
            return any(term in user_lower for term in search_terms)

    def search_web(self, query: str, max_results: int = 3) -> str:
        """Search using SearXNG and extract key information"""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; SmartChatbot/1.0)"}
            params = {
                "q": query,
                "format": "json",
                "safesearch": "0"
            }
            
            response = requests.get(f"{self.searxng_url}/search", 
                                  params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            results = response.json().get("results", [])
            
            if not results:
                return "No current search results found."
            
            # Extract useful information from top results
            search_summary = []
            for i, result in enumerate(results[:max_results]):
                title = result.get("title", "")
                content = result.get("content", "")
                
                # Clean and format result - focus on the actual content
                if content:
                    # Take first 150 chars of content and clean it
                    clean_content = content[:150].strip()
                    # Remove common web cruft
                    clean_content = clean_content.replace("...", "")
                    search_summary.append(f"• {title}: {clean_content}")
                elif title:
                    search_summary.append(f"• {title}")
            
            return "\n".join(search_summary)
            
        except requests.exceptions.RequestException as e:
            return f"Search currently unavailable: {e}"
        except Exception as e:
            return f"Search error: {e}"

    def generate_response(self, user_input: str, use_search: bool = False) -> str:
        """Generate response with optional web search"""
        
        # Build conversation context
        context = self._build_context()
        
        # Get search info if needed
        search_info = ""
        if use_search:
            print("Searching web...")
            search_info = self.search_web(user_input)
        
        # Build prompt with better search integration
        if search_info and search_info != "No current search results found.":
            prompt = f"""{context}Current web search results for reference:
{search_info}

User: {user_input}
Assistant:"""
        else:
            prompt = f"""{context}User: {user_input}
Assistant:"""
        
        # Generate response
        start_time = time.time()
        
        try:
            tokens = self.model.tokenize(prompt)
            response_tokens = []
            
            # Generate with better stopping logic
            for token in self.model.generate(tokens):
                response_tokens.append(token)
                
                # Check for natural stopping points
                current_text = self.model.detokenize(response_tokens)
                
                # Stop if we hit conversation markers
                stop_markers = ["\nUser:", "\nHuman:", "User:", "Human:"]
                should_stop = False
                
                for marker in stop_markers:
                    if marker in current_text:
                        # Remove everything from the marker onwards
                        current_text = current_text.split(marker)[0]
                        should_stop = True
                        break
                
                if should_stop:
                    response_tokens = self.model.tokenize(current_text)
                    break
                
                # Safety limit to prevent infinite generation  
                if len(response_tokens) > 200:
                    break
            
            response = self.model.detokenize(response_tokens).strip()
            
            # Clean up response
            if response.startswith("Assistant:"):
                response = response[10:].strip()
            
            # Remove any trailing conversation markers
            for marker in ["\nUser", "\nHuman", "User", "Human"]:
                if response.endswith(marker):
                    response = response[:-len(marker)].strip()
            
            generation_time = time.time() - start_time
            print(f"[Generated in {generation_time:.2f}s, {len(response_tokens)} tokens]")
            
            return response
            
        except Exception as e:
            return f"Sorry, I encountered an error: {e}"

    def _build_context(self) -> str:
        """Build conversation context with anti-hallucination measures"""
        if not self.conversation:
            return "You are a helpful AI assistant. Respond naturally and conversationally.\n"
        
        context_parts = ["Recent conversation:"]
        for exchange in self.conversation[-self.max_history:]:
            # Sanitize conversation history to prevent model confusion
            user_msg = exchange['user'].strip()
            bot_msg = exchange['bot'].strip()
            
            # Ensure clean formatting
            context_parts.append(f"User: {user_msg}")
            context_parts.append(f"Assistant: {bot_msg}")
        
        context_parts.append("")  # Empty line before current exchange
        return "\n".join(context_parts)

    def chat(self):
        """Main chat loop"""
        print("\nChatbot ready! Type 'quit' to exit, 'clear' to reset.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation = []
                    print("Conversation cleared.\n")
                    continue
                
                if not user_input:
                    continue
                
                # Decide if search is needed
                context = self._build_context()
                needs_search = self.should_search(user_input, context)
                
                if needs_search:
                    print("[Searching for current information...]")
                
                # Generate response
                response = self.generate_response(user_input, needs_search)
                print(f"Bot: {response}\n")
                
                # Update conversation
                self.conversation.append({
                    'user': user_input,
                    'bot': response
                })
                
                # Trim history
                if len(self.conversation) > self.max_history:
                    self.conversation = self.conversation[-self.max_history:]
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    # Initialize chatbot
    chatbot = SmartChatbot()
    
    # Start chatting
    chatbot.chat()

if __name__ == "__main__":
    main()