"""
Conversational QA Service with conversation history and context management
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class Message:
    """Single message in conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSession:
    """Conversation session with history"""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to conversation"""
        msg = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(msg)
        self.updated_at = datetime.now()
    
    def get_history(self, last_n: Optional[int] = None) -> List[Message]:
        """Get conversation history"""
        if last_n:
            return self.messages[-last_n:]
        return self.messages
    
    def get_context_string(self, last_n: int = 5) -> str:
        """Get conversation context as formatted string"""
        history = self.get_history(last_n)
        context_parts = []
        for msg in history:
            prefix = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{prefix}: {msg.content}")
        return "\n".join(context_parts)


class ConversationalQA:
    """
    Conversational QA with session management and follow-up handling
    """
    
    def __init__(self, adaptive_qa, hybrid_retrieval):
        """
        Initialize conversational QA
        
        Args:
            adaptive_qa: AdaptiveQAService instance
            hybrid_retrieval: HybridRetriever instance
        """
        self.adaptive_qa = adaptive_qa
        self.hybrid_retrieval = hybrid_retrieval
        self.sessions: Dict[str, ConversationSession] = {}
    
    def create_session(self, session_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """Create new conversation session"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = ConversationSession(
            session_id=session_id,
            metadata=metadata or {}
        )
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get conversation session"""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete conversation session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def reformulate_question(self, question: str, conversation_context: str) -> str:
        """
        Reformulate follow-up question with conversation context
        
        Simple approach: prepend context if question seems like follow-up
        """
        follow_up_indicators = [
            "nó", "họ", "đó", "kia", "này", "gì nữa", "còn", 
            "it", "that", "this", "those", "what else", "and", "also"
        ]
        
        question_lower = question.lower()
        is_follow_up = any(indicator in question_lower for indicator in follow_up_indicators)
        
        if is_follow_up and conversation_context:
            # Prepend recent context
            return f"Context: {conversation_context}\n\nQuestion: {question}"
        
        return question
    
    def chat(
        self,
        session_id: str,
        question: str,
        top_k: int = 5,
        use_context: bool = True,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Chat with conversational context
        
        Args:
            session_id: Conversation session ID
            question: User question
            top_k: Number of chunks to retrieve
            use_context: Use conversation history for context
            source_language: Source language override
            target_language: Target language for answer
            
        Returns:
            Dict with answer, sources, and session info
        """
        # Get or create session
        if not session_id or session_id == "":
            session_id = str(uuid.uuid4())
        
        session = self.get_session(session_id)
        if session is None:
            session_id = self.create_session(session_id)
            session = self.get_session(session_id)
        
        # Get conversation context
        conversation_context = ""
        if use_context and len(session.messages) > 0:
            conversation_context = session.get_context_string(last_n=3)
            
        # Reformulate question if needed
        reformulated_question = self.reformulate_question(question, conversation_context)
        
        # Retrieve relevant contexts
        search_results = self.hybrid_retrieval.hybrid_search(
            query=reformulated_question,
            top_k=top_k,
            alpha=0.5
        )
        
        if not search_results:
            answer = "Tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
            session.add_message("user", question)
            session.add_message("assistant", answer, {"no_results": True})
            
            return {
                "session_id": session_id,
                "answer": answer,
                "sources": [],
                "confidence": 0.0,
                "reformulated_question": reformulated_question,
                "used_context": use_context and len(session.messages) > 2,
                "conversation_length": len(session.messages)
            }
        
        # Combine contexts
        context = " ".join([r["text"] for r in search_results[:3]])
        
        # Get answer with adaptive QA
        result = self.adaptive_qa.answer(
            question=reformulated_question,
            context=context,
            source_language=source_language,
            target_language=target_language
        )
        
        # Add to conversation history
        session.add_message("user", question, {
            "reformulated": reformulated_question,
            "top_k": top_k
        })
        session.add_message("assistant", result.answer, {
            "confidence": result.score,
            "sources_count": len(search_results),
            "translated": result.translated
        })
        
        return {
            "session_id": session_id,
            "answer": result.answer,
            "sources": [
                {
                    "text": src["text"],
                    "score": src["score"],
                    "metadata": src.get("metadata", {})
                }
                for src in search_results
            ],
            "confidence": result.score,
            "source_language": result.source_language,
            "target_language": result.target_language,
            "translated": result.translated,
            "reformulated_question": reformulated_question,
            "used_context": use_context and len(session.messages) > 2,
            "conversation_length": len(session.messages)
        }
    
    def get_conversation_history(
        self,
        session_id: str,
        last_n: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get conversation history for a session"""
        session = self.get_session(session_id)
        if session is None:
            return None
        
        messages = session.get_history(last_n)
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata
            }
            for msg in messages
        ]
    
    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history but keep session"""
        session = self.get_session(session_id)
        if session:
            session.messages = []
            session.updated_at = datetime.now()
            return True
        return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        return [
            {
                "session_id": session.session_id,
                "message_count": len(session.messages),
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata
            }
            for session in self.sessions.values()
        ]


def create_conversational_qa(adaptive_qa, hybrid_retrieval) -> ConversationalQA:
    """Factory function to create conversational QA service"""
    return ConversationalQA(
        adaptive_qa=adaptive_qa,
        hybrid_retrieval=hybrid_retrieval
    )
