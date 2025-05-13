import time
import threading
import uuid
from typing import Dict, Any, Optional, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langgraph.checkpoint.memory import MemorySaver


class SafeMemorySaver:
    """Wrapper around LangGraph's MemorySaver to handle API changes safely."""
    def __init__(self):
        self.memory_saver = MemorySaver()
        self.api_version = self._detect_api_version()
        
    def _detect_api_version(self):
        import inspect
        sig = inspect.signature(self.memory_saver.put)
        param_names = list(sig.parameters.keys())
        print(f"Detected MemorySaver API parameters: {param_names}")
        return len(param_names)
    
    def put(self, config, state, metadata, new_versions):
        try:
            if self.api_version == 4:
                return self.memory_saver.put(config, state, metadata, new_versions)
            elif self.api_version == 3:
                return self.memory_saver.put(config, state, metadata)
            else:
                print("Using fallback API approach")
                try:
                    return self.memory_saver.put(config["configurable"]["thread_id"], state)
                except Exception as e1:
                    print(f"First fallback failed: {e1}")
                    try:
                        return self.memory_saver.put(config, state)
                    except Exception as e2:
                        print(f"Second fallback failed: {e2}")
                        return None
        except Exception as e:
            print(f"Error in SafeMemorySaver.put: {e}")
            return None
    
    def delete(self, session_id):
        try:
            if self.api_version >= 2:
                config = {"configurable": {"thread_id": session_id}}
                return self.memory_saver.delete(config)
            else:
                print("Using fallback delete approach")
                try:
                    return self.memory_saver.delete(session_id)
                except Exception as e1:
                    print(f"First delete fallback failed: {e1}")
                    try:
                        config = {"configurable": {"thread_id": session_id}}
                        return self.memory_saver.delete(config)
                    except Exception as e2:
                        print(f"Second delete fallback failed: {e2}")
                        return None
        except Exception as e:
            print(f"Error in SafeMemorySaver.delete: {e}")
            return None


class SessionMemory:
    def __init__(self, max_history_length: int = 20):
        self.max_history_length = max_history_length
        self.created_at = time.time()
        self.last_accessed_at = time.time()
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history",
            input_key="input",
            output_key="output"
        )
    
    def add_user_message(self, message: str):
        self.memory.chat_memory.add_user_message(message)
        self.last_accessed_at = time.time()
    
    def add_ai_message(self, message: str):
        self.memory.chat_memory.add_ai_message(message)
        self.last_accessed_at = time.time()
    
    def get_chat_history(self) -> List[BaseMessage]:
        return self.memory.chat_memory.messages[-self.max_history_length:]
    
    def get_messages(self) -> List[BaseMessage]:
        return self.memory.chat_memory.messages
    
    def get_conversation_buffer(self) -> str:
        messages = self.get_chat_history()
        history = []
        for message in messages:
            prefix = "Human: " if isinstance(message, HumanMessage) else "Assistant: "
            history.append(f"{prefix}{message.content}")
        return "\n".join(history)


class ChatbotMemoryManager:
    def __init__(self, 
                 expiry_minutes: int = 30, 
                 max_sessions: int = 100, 
                 max_history_length: int = 20):
        self.sessions: Dict[str, SessionMemory] = {}
        self.memory_saver = SafeMemorySaver()
        self.expiry_seconds = expiry_minutes * 60
        self.max_sessions = max_sessions
        self.max_history_length = max_history_length
        
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired_sessions, daemon=True)
        self.cleanup_thread.start()
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        if len(self.sessions) >= self.max_sessions:
            self._remove_oldest_session()
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        session_memory = SessionMemory(self.max_history_length)
        self.sessions[session_id] = session_memory
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionMemory]:
        session = self.sessions.get(session_id)
        if session:
            session.last_accessed_at = time.time()
            return session
        return None
    
    def add_user_message(self, session_id: str, message: str):
        session = self.get_session(session_id)
        if not session:
            session_id = self.create_session(session_id)
            session = self.get_session(session_id)
        
        session.add_user_message(message)
        self._update_langgraph_checkpoint(session_id, session.get_messages())
    
    def add_ai_message(self, session_id: str, message: str):
        session = self.get_session(session_id)
        if not session:
            session_id = self.create_session(session_id)
            session = self.get_session(session_id)
        
        session.add_ai_message(message)
        self._update_langgraph_checkpoint(session_id, session.get_messages())
    
    def get_chat_history(self, session_id: str) -> str:
        session = self.get_session(session_id)
        return session.get_conversation_buffer() if session else ""
    
    def get_messages(self, session_id: str) -> List[BaseMessage]:
        session = self.get_session(session_id)
        return session.get_messages() if session else []
    
    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            self._clear_langgraph_checkpoint(session_id)
    
    def _remove_oldest_session(self):
        if not self.sessions:
            return
        
        oldest_session_id = min(
            self.sessions.keys(), 
            key=lambda sid: self.sessions[sid].last_accessed_at
        )
        self.clear_session(oldest_session_id)
    
    def _cleanup_expired_sessions(self):
        while True:
            current_time = time.time()
            expired_sessions = [
                sid for sid, session in self.sessions.items()
                if current_time - session.last_accessed_at > self.expiry_seconds
            ]
            
            for session_id in expired_sessions:
                self.clear_session(session_id)
            
            if expired_sessions:
                print(f"Cleaned up {len(expired_sessions)} expired chat sessions")
                print(f"Active sessions: {len(self.sessions)}")
            
            time.sleep(300)  # 5 minutes
    
    def _update_langgraph_checkpoint(self, session_id: str, messages: List[BaseMessage]):
        state = {"messages": messages}
        config = {
            "configurable": {"thread_id": session_id}
        }
        metadata = {"session_id": session_id}
        new_versions = {"version": 1}

        try:
            self.memory_saver.put(config, state, metadata, new_versions)
        except Exception as e:
            print(f"Error updating checkpoint: {e}")
    
    def _clear_langgraph_checkpoint(self, session_id: str):
        config = {
            "configurable": {"thread_id": session_id}
        }
        try:
            self.memory_saver.delete(config)
        except Exception as e:
            print(f"Error clearing checkpoint: {e}")
