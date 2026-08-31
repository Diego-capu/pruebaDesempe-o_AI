import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, AlertCircle, Clock, DollarSign, Zap, Bot, User } from 'lucide-react';

export interface RetrievedChunk {
  text: string;
  source: string;
  similarity: number;
}

export interface EscalationDetails {
  stage: string;
  reason?: string;
  ticket_id?: string;
  max_similarity_score?: number;
}

export interface AgentMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  isStreaming?: boolean;
  escalated?: boolean;
  escalationDetails?: EscalationDetails | null;
  sources?: string[];
  retrievedChunks?: RetrievedChunk[];
  suggestedChips?: string[];
  latencyMs?: number;
  estimatedCostUsd?: number;
  cached?: boolean;
  timestamp?: string;
}

interface AgentChatProps {
  messages: AgentMessage[];
  onSendMessage: (query: string) => Promise<void>;
  isLoading?: boolean;
  initialSuggestions?: string[];
  onSuggestionClick?: (suggestion: string) => void;
  className?: string;
}

export const AgentChat: React.FC<AgentChatProps> = ({
  messages,
  onSendMessage,
  isLoading = false,
  initialSuggestions = [
    "What undergraduate degree programs are offered and what are the class schedules?",
    "How much is full-time undergraduate tuition per semester?",
    "Are there any scholarships for women in STEM?",
    "What AWS or Cisco industry certifications are included in the curriculum?"
  ],
  onSuggestionClick,
  className = ""
}) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    const query = input.trim();
    setInput('');
    onSendMessage(query);
  };

  const handleChipClick = (chipText: string) => {
    if (onSuggestionClick) {
      onSuggestionClick(chipText);
    } else {
      onSendMessage(chipText);
    }
  };

  return (
    <div className={`flex flex-col h-full bg-neutral-950/25 hover:bg-neutral-950/80 focus-within:bg-neutral-950/80 border border-neutral-800/40 hover:border-neutral-800/90 focus-within:border-neutral-800/90 rounded-2xl overflow-hidden shadow-2xl backdrop-blur-sm hover:backdrop-blur-2xl focus-within:backdrop-blur-2xl transition-all duration-500 group ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-neutral-800/40 group-hover:border-neutral-800/80 group-focus-within:border-neutral-800/80 bg-neutral-900/30 group-hover:bg-neutral-900/70 group-focus-within:bg-neutral-900/70 backdrop-blur-md transition-all duration-300">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-500 to-teal-400 flex items-center justify-center text-white font-bold shadow-md shadow-indigo-500/20">
            <Bot className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-neutral-100 tracking-tight">Miku AI</h2>
            <p className="text-xs text-neutral-400">Grounded RAG Pipeline & Real-Time SSE</p>
          </div>
        </div>
        <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-teal-950/40 border border-teal-500/30 text-teal-400 text-xs font-medium backdrop-blur-md">
          <span className="w-2 h-2 rounded-full bg-teal-400 animate-pulse shadow-sm shadow-teal-400/50" />
          Online
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex gap-3 max-w-[88%] ${msg.role === 'user' ? 'ml-auto flex-row-reverse' : 'mr-auto'}`}
          >
            {/* Avatar */}
            <div
              className={`w-7 h-7 rounded-md flex-shrink-0 flex items-center justify-center text-xs font-semibold ${
                msg.role === 'user'
                  ? 'bg-neutral-800 text-neutral-200 border border-neutral-700'
                  : 'bg-indigo-600/30 text-indigo-300 border border-indigo-500/30'
              }`}
            >
              {msg.role === 'user' ? <User className="w-3.5 h-3.5" /> : <Bot className="w-3.5 h-3.5" />}
            </div>

            {/* Bubble */}
            <div
              className={`rounded-2xl px-4 py-3 text-sm leading-relaxed transition-all duration-300 ${
                msg.role === 'user'
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20'
                  : 'bg-neutral-900/60 group-hover:bg-neutral-900/90 group-focus-within:bg-neutral-900/90 text-neutral-200 border border-neutral-800/50 group-hover:border-neutral-800 shadow-sm backdrop-blur-md'
              }`}
            >
              <div className="whitespace-pre-wrap">
                {msg.content}
                {msg.isStreaming && (
                  <span className="inline-block w-1.5 h-3.5 bg-teal-400 ml-1 animate-pulse align-middle" />
                )}
              </div>

              {/* Human Escalation Alert Box */}
              {msg.escalated && msg.escalationDetails && (
                <div className="mt-3 p-3 bg-rose-950/40 border border-rose-500/30 rounded-xl text-rose-200 text-xs flex flex-col gap-1 backdrop-blur-md">
                  <div className="flex items-center gap-1.5 font-semibold text-rose-400">
                    <AlertCircle className="w-4 h-4" />
                    <span>Human Escalation Triggered</span>
                  </div>
                  <div className="text-neutral-300">
                    Stage: <span className="text-neutral-100">{msg.escalationDetails.stage}</span>
                  </div>
                  {msg.escalationDetails.ticket_id && (
                    <div className="font-mono text-rose-300 font-semibold">
                      Ticket ID: {msg.escalationDetails.ticket_id}
                    </div>
                  )}
                </div>
              )}

              {/* Metadata Badge */}
              {msg.role === 'assistant' && (msg.latencyMs !== undefined || msg.estimatedCostUsd !== undefined) && (
                <div className="mt-2.5 pt-2 border-t border-neutral-800/60 flex items-center gap-3 text-[11px] font-mono text-neutral-400">
                  {msg.latencyMs !== undefined && (
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3 text-neutral-400" />
                      {msg.latencyMs}ms
                    </span>
                  )}
                  {msg.estimatedCostUsd !== undefined && (
                    <span className="flex items-center gap-1">
                      <DollarSign className="w-3 h-3 text-neutral-400" />
                      ${msg.estimatedCostUsd.toFixed(5)}
                    </span>
                  )}
                  {msg.cached && (
                    <span className="flex items-center gap-1 text-amber-400 font-medium">
                      <Zap className="w-3 h-3" />
                      Cached
                    </span>
                  )}
                </div>
              )}

              {/* Interactive Quick Reply Chips */}
              {msg.suggestedChips && msg.suggestedChips.length > 0 && (
                <div className="mt-3.5 flex flex-wrap gap-1.5">
                  {msg.suggestedChips.map((chip, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleChipClick(chip)}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-neutral-800/80 hover:bg-neutral-700/90 text-neutral-300 hover:text-white border border-neutral-700/50 hover:border-indigo-500/50 transition-all duration-150 cursor-pointer shadow-sm active:scale-95"
                    >
                      <Sparkles className="w-3 h-3 text-teal-400" />
                      <span>{chip}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Quick Prompt Bar (Top of Input) */}
      {messages.length <= 1 && (
        <div className="px-5 py-2.5 border-t border-neutral-800/40 group-hover:border-neutral-800/60 group-focus-within:border-neutral-800/60 bg-neutral-900/20 group-hover:bg-neutral-900/50 group-focus-within:bg-neutral-900/50 backdrop-blur-md flex items-center gap-2 overflow-x-auto no-scrollbar transition-all duration-300">
          {initialSuggestions.map((suggestion, idx) => (
            <button
              key={idx}
              onClick={() => handleChipClick(suggestion)}
              className="flex-shrink-0 px-3 py-1 rounded-full text-xs bg-neutral-900/80 border border-neutral-800/70 hover:border-neutral-700 text-neutral-400 hover:text-neutral-200 transition-colors"
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-neutral-800/40 group-hover:border-neutral-800/80 group-focus-within:border-neutral-800/80 bg-neutral-900/30 group-hover:bg-neutral-900/70 group-focus-within:bg-neutral-900/70 backdrop-blur-md transition-all duration-300">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about admissions, programs, tuition, schedules..."
            disabled={isLoading}
            className="flex-1 bg-neutral-950/70 focus:bg-neutral-950/95 border border-neutral-800/80 focus:border-indigo-500 rounded-xl px-4 py-2.5 text-sm text-neutral-100 placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 transition-all font-sans"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium text-sm flex items-center gap-2 transition-all shadow-md shadow-indigo-600/25 active:scale-95 cursor-pointer"
          >
            <span>Send</span>
            <Send className="w-3.5 h-3.5" />
          </button>
        </div>
      </form>
    </div>
  );
};

export default AgentChat;
