import streamlit as st
import os
import pdfplumber
import io
import re
from typing import List, Optional, Dict, Tuple
from collections import Counter
import math

# Import error handling with version compatibility
try:
    import openai
    OPENAI_AVAILABLE = True
    
    # Check OpenAI version more reliably
    try:
        # Try to import the new client class
        from openai import OpenAI
        OPENAI_VERSION = "new"  # v1.x
    except ImportError:
        OPENAI_VERSION = "old"  # v0.x
        
except ImportError:
    OPENAI_AVAILABLE = False
    OPENAI_VERSION = None
    st.error("OpenAI package not installed. Please install it with: pip install openai")

st.set_page_config(page_title="Ask my PDF - Smart Edition", layout="wide")

# Advanced text processing functions
def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove page numbers and headers/footers patterns
    text = re.sub(r'\n\d+\n', ' ', text)
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    return text

def extract_text_from_pdf(uploaded_file, max_chunks: int = 10) -> List[Dict]:
    """Extract text chunks with metadata from uploaded PDF file."""
    try:
        text_chunks = []
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_chunks:
                    break
                
                # Extract text
                text = page.extract_text()
                if text and len(text.strip()) > 50:  # Only meaningful chunks
                    cleaned_text = clean_text(text)
                    
                    # Extract tables if present
                    tables = page.extract_tables()
                    table_text = ""
                    if tables:
                        for table in tables:
                            table_text += "\n" + "\n".join([" | ".join([cell or "" for cell in row]) for row in table])
                    
                    chunk_data = {
                        'page': i + 1,
                        'text': cleaned_text,
                        'tables': table_text,
                        'length': len(cleaned_text),
                        'word_count': len(cleaned_text.split())
                    }
                    text_chunks.append(chunk_data)
        
        return text_chunks
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return []

def calculate_relevance_score(text: str, query: str) -> float:
    """Calculate relevance score using TF-IDF-like approach."""
    # Convert to lowercase for comparison
    text_lower = text.lower()
    query_lower = query.lower()
    
    # Split into words
    text_words = re.findall(r'\b\w+\b', text_lower)
    query_words = re.findall(r'\b\w+\b', query_lower)
    
    if not query_words or not text_words:
        return 0.0
    
    # Calculate term frequency
    text_word_count = Counter(text_words)
    total_words = len(text_words)
    
    score = 0.0
    matched_terms = 0
    
    for query_word in query_words:
        if len(query_word) > 2:  # Ignore short words
            # Exact match
            if query_word in text_word_count:
                tf = text_word_count[query_word] / total_words
                score += tf * 10  # Boost exact matches
                matched_terms += 1
            
            # Partial match (contains)
            partial_matches = sum(1 for word in text_words if query_word in word and len(word) > len(query_word))
            if partial_matches > 0:
                score += (partial_matches / total_words) * 5
                matched_terms += 0.5
    
    # Boost score based on query coverage
    coverage_bonus = matched_terms / len(query_words)
    score *= (1 + coverage_bonus)
    
    # Proximity bonus - check if query words appear close to each other
    query_phrase = " ".join(query_words)
    if query_phrase in text_lower:
        score *= 2  # Strong boost for exact phrase match
    
    return score

def find_relevant_chunks(text_chunks: List[Dict], query: str, top_k: int = 3) -> List[Dict]:
    """Find most relevant text chunks for the query."""
    scored_chunks = []
    
    for chunk in text_chunks:
        # Score main text
        text_score = calculate_relevance_score(chunk['text'], query)
        
        # Score tables if present
        table_score = 0
        if chunk['tables']:
            table_score = calculate_relevance_score(chunk['tables'], query) * 0.8
        
        total_score = text_score + table_score
        
        if total_score > 0:
            chunk_with_score = chunk.copy()
            chunk_with_score['relevance_score'] = total_score
            scored_chunks.append(chunk_with_score)
    
    # Sort by relevance score
    scored_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return scored_chunks[:top_k]

def create_smart_context(relevant_chunks: List[Dict], query: str) -> str:
    """Create intelligent context from relevant chunks."""
    if not relevant_chunks:
        return "No relevant content found in the PDF."
    
    context_parts = []
    
    for i, chunk in enumerate(relevant_chunks):
        page_info = f"[Page {chunk['page']}]"
        
        # Extract most relevant sentences
        sentences = re.split(r'[.!?]+', chunk['text'])
        relevant_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 20:  # Meaningful sentences
                sentence_score = calculate_relevance_score(sentence, query)
                if sentence_score > 0:
                    relevant_sentences.append((sentence.strip(), sentence_score))
        
        # Sort sentences by relevance and take top ones
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in relevant_sentences[:3]]
        
        if top_sentences:
            chunk_text = ". ".join(top_sentences)
            
            # Add table data if relevant
            if chunk['tables'] and calculate_relevance_score(chunk['tables'], query) > 0:
                chunk_text += f"\n\nTable data from page {chunk['page']}:\n{chunk['tables'][:300]}..."
            
            context_parts.append(f"{page_info}\n{chunk_text}")
    
    return "\n\n---\n\n".join(context_parts)

def generate_smart_prompt(query: str, context: str, document_summary: str = "") -> str:
    """Generate an intelligent prompt for better responses."""
    
    # Detect query type
    query_lower = query.lower()
    is_definition = any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning'])
    is_howto = any(word in query_lower for word in ['how to', 'how do', 'steps', 'process'])
    is_comparison = any(word in query_lower for word in ['difference', 'compare', 'versus', 'vs'])
    is_list = any(word in query_lower for word in ['list', 'types', 'kinds', 'examples'])
    
    # Customize prompt based on query type
    if is_definition:
        instruction = "Provide a clear, comprehensive definition based on the document content."
    elif is_howto:
        instruction = "Explain the process step-by-step based on the information provided."
    elif is_comparison:
        instruction = "Compare and contrast the elements mentioned, highlighting key differences."
    elif is_list:
        instruction = "Provide a well-organized list with explanations for each item."
    else:
        instruction = "Provide a detailed, accurate answer based on the document content."
    
    prompt = f"""You are an expert document analyst. Answer the question based ONLY on the provided document content.

DOCUMENT CONTENT:
{context}

QUESTION: {query}

INSTRUCTIONS:
{instruction}

GUIDELINES:
- Use only information from the provided document content
- If the answer isn't in the document, say so clearly
- Cite page numbers when possible
- Be specific and detailed
- Structure your answer clearly
- If there are tables or data, incorporate them appropriately

ANSWER:"""
    
    return prompt

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """Extract key terms from text."""
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_freq = Counter(words)
    
    # Filter out common words
    stop_words = {'the', 'and', 'are', 'for', 'with', 'this', 'that', 'from', 'they', 'have', 'was', 'been', 'their', 'said', 'each', 'which', 'can', 'all', 'but', 'not', 'you', 'any', 'had', 'her', 'him', 'has', 'how', 'its', 'our', 'out', 'day', 'get', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'man', 'end', 'few', 'got', 'his', 'let', 'put', 'say', 'she', 'too', 'use'}
    
    filtered_words = {word: freq for word, freq in word_freq.items() if word not in stop_words and len(word) > 3}
    
    return [word for word, freq in Counter(filtered_words).most_common(top_k)]

# Improved OpenAI API handling
def test_openai_connection(api_key: str, model: str) -> tuple[bool, str, object]:
    """Test OpenAI connection and return client if successful."""
    try:
        if OPENAI_VERSION == "new":
            # OpenAI v1.x
            client = OpenAI(api_key=api_key)
            
            # Test with a simple call
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True, "Connection successful", client
            
        else:
            # OpenAI v0.x (legacy)
            openai.api_key = api_key
            
            # Test with a simple call
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True, "Connection successful", "legacy"
            
    except Exception as e:
        error_msg = str(e)
        
        # Provide specific error messages
        if "invalid_api_key" in error_msg or "Incorrect API key" in error_msg:
            return False, "Invalid API key. Please check your OpenAI API key.", None
        elif "insufficient_quota" in error_msg:
            return False, "Insufficient quota. Please check your OpenAI billing.", None
        elif "model_not_found" in error_msg:
            return False, f"Model '{model}' not found. Please check model availability.", None
        elif "rate_limit" in error_msg:
            return False, "Rate limit exceeded. Please try again later.", None
        else:
            return False, f"Connection failed: {error_msg}", None

def make_openai_request(client, model: str, messages: list, max_tokens: int = 1500) -> tuple[bool, str, dict]:
    """Make OpenAI API request with proper error handling."""
    try:
        if OPENAI_VERSION == "new" and client != "legacy":
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            usage_info = {
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
            
            return True, answer, usage_info
            
        else:
            # Legacy OpenAI v0.x
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            usage_info = {
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
            
            return True, answer, usage_info
            
    except Exception as e:
        error_msg = str(e)
        
        # Handle specific errors
        if "rate_limit" in error_msg:
            return False, "Rate limit exceeded. Please try again in a few minutes.", {}
        elif "insufficient_quota" in error_msg:
            return False, "Insufficient quota. Please check your OpenAI billing.", {}
        elif "context_length" in error_msg:
            return False, "Context too long. Please reduce the number of pages or context chunks.", {}
        elif "invalid_request" in error_msg:
            return False, "Invalid request. Please try a different question or model.", {}
        else:
            return False, f"API Error: {error_msg}", {}

# Sidebar setup
st.sidebar.title("🧠 Smart PDF Q&A")
st.sidebar.markdown("**version 2.0 - Enhanced AI**")
st.sidebar.markdown("Built by **Saidatta Dasari**")
st.sidebar.markdown("Advanced PDF Question Answering with intelligent text processing and semantic search.")
st.sidebar.markdown("---")

st.sidebar.title("🔧 Advanced Settings")
max_pages = st.sidebar.number_input("Max Pages to Process", min_value=1, max_value=50, value=10, step=1)
relevance_threshold = st.sidebar.slider("Relevance Threshold", 0.1, 2.0, 0.5, 0.1)
max_context_chunks = st.sidebar.number_input("Max Context Chunks", min_value=1, max_value=10, value=3, step=1)

st.sidebar.title("🎯 Query Enhancement")
use_smart_context = st.sidebar.checkbox("🧠 Smart Context Selection", value=True)
use_semantic_search = st.sidebar.checkbox("🔍 Enhanced Relevance Scoring", value=True)
show_relevance_scores = st.sidebar.checkbox("📊 Show Relevance Scores", value=False)

# Header
st.title("🧠 Smart PDF Q&A System")
st.caption("Advanced question answering with intelligent text processing and semantic search")

# Interface mode selection
mode = st.radio("Choose your mode", ["🔐 OpenAI GPT Analysis", "🌐 Local Smart Processing"])

# OpenAI API handling
openai_client = None
if mode == "🔐 OpenAI GPT Analysis":
    if not OPENAI_AVAILABLE:
        st.error("OpenAI package is not available. Please install it first.")
        st.code("pip install openai", language="bash")
    else:
        st.info("💡 **Need an API key?** Get one from [OpenAI's website](https://platform.openai.com/api-keys)")
        
        api_key = st.text_input(
            "Enter your OpenAI API Key", 
            type="password", 
            help="Your API key will not be stored and is only used for this session"
        )
        
        model_choice = st.selectbox(
            "Choose Model", 
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"],
            help="GPT-4 models provide better analysis but cost more"
        )
        
        if api_key:
            # Test connection
            with st.spinner("🔍 Testing API connection..."):
                success, message, client = test_openai_connection(api_key, model_choice)
                
            if success:
                st.success(f"✅ {message}")
                openai_client = client
                
                # Show additional info
                st.info(f"🤖 Using {model_choice} | Version: OpenAI {OPENAI_VERSION}")
                
            else:
                st.error(f"❌ {message}")
                
                # Provide helpful suggestions
                with st.expander("🔧 Troubleshooting"):
                    st.markdown("""
                    **Common Issues & Solutions:**
                    
                    1. **Invalid API Key**: 
                       - Check your API key at [OpenAI Platform](https://platform.openai.com/api-keys)
                       - Make sure there are no extra spaces
                    
                    2. **Insufficient Quota**: 
                       - Add billing information to your OpenAI account
                       - Check your usage limits
                    
                    3. **Model Not Found**: 
                       - Some models require special access
                       - Try using `gpt-3.5-turbo` first
                    
                    4. **Rate Limits**: 
                       - Wait a few minutes before trying again
                       - Consider upgrading your OpenAI plan
                    """)
                
else:
    st.info("🌐 Using local smart processing with advanced text analysis")

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload your PDF file", type=["pdf"], help="Select a PDF file to analyze")

if uploaded_file is not None:
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    # Process the PDF
    with st.spinner("🔍 Analyzing PDF with smart processing..."):
        text_chunks = extract_text_from_pdf(uploaded_file, max_chunks=max_pages)
    
    if text_chunks:
        # Document statistics
        total_pages = len(text_chunks)
        total_words = sum(chunk['word_count'] for chunk in text_chunks)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Pages Processed", total_pages)
        with col2:
            st.metric("📝 Total Words", f"{total_words:,}")
        with col3:
            st.metric("⚡ Processing Mode", "Smart AI" if use_smart_context else "Basic")
        
        # Extract and show document keywords
        all_text = " ".join([chunk['text'] for chunk in text_chunks])
        keywords = extract_keywords(all_text, 15)
        
        with st.expander("🔍 Document Keywords & Preview"):
            st.write("**Key Topics:**")
            st.write(", ".join(keywords))
            st.write("**Sample Content:**")
            preview_text = text_chunks[0]['text'][:600] + "..." if len(text_chunks[0]['text']) > 600 else text_chunks[0]['text']
            st.text(preview_text)
        
        # Ask question
        st.markdown("### 💬 Ask Your Question")
        
        # Suggested questions based on keywords
        suggested_questions = [
            f"What is {keywords[0]}?" if keywords else "What is this document about?",
            f"How does {keywords[1]} work?" if len(keywords) > 1 else "What are the main points?",
            f"What are the types of {keywords[2]}?" if len(keywords) > 2 else "Can you summarize this?",
        ]
        
        st.write("**💡 Suggested questions:**")
        for i, suggestion in enumerate(suggested_questions):
            if st.button(f"📝 {suggestion}", key=f"suggestion_{i}"):
                st.session_state.query_input = suggestion
        
        query = st.text_input(
            "Your question:", 
            value=st.session_state.get('query_input', ''),
            placeholder="Ask anything about the document...",
            key="main_query"
        )
        
        if query:
            with st.spinner("🤔 Finding relevant information..."):
                if use_smart_context:
                    # Smart processing
                    relevant_chunks = find_relevant_chunks(text_chunks, query, max_context_chunks)
                    context = create_smart_context(relevant_chunks, query)
                    
                    if show_relevance_scores and relevant_chunks:
                        st.write("**🎯 Relevance Scores:**")
                        for chunk in relevant_chunks:
                            st.write(f"Page {chunk['page']}: {chunk['relevance_score']:.2f}")
                else:
                    # Basic processing
                    context = " ".join([chunk['text'] for chunk in text_chunks[:max_context_chunks]])
                    relevant_chunks = text_chunks[:max_context_chunks]
                
                if not context or context == "No relevant content found in the PDF.":
                    st.warning("⚠️ No relevant content found. Try rephrasing your question.")
                else:
                    # Generate response
                    if mode == "🔐 OpenAI GPT Analysis" and openai_client:
                        with st.spinner("🤖 Generating AI response..."):
                            smart_prompt = generate_smart_prompt(query, context)
                            
                            messages = [{"role": "user", "content": smart_prompt}]
                            
                            success, answer, usage_info = make_openai_request(
                                openai_client, 
                                model_choice, 
                                messages, 
                                max_tokens=1500
                            )
                            
                            if success:
                                st.markdown("### 🤖 AI Analysis")
                                st.write(answer)
                                
                                # Show usage statistics
                                with st.expander("📊 Response Details"):
                                    if usage_info:
                                        st.write(f"**Total tokens:** {usage_info.get('total_tokens', 'N/A')}")
                                        st.write(f"**Prompt tokens:** {usage_info.get('prompt_tokens', 'N/A')}")
                                        st.write(f"**Completion tokens:** {usage_info.get('completion_tokens', 'N/A')}")
                                    st.write(f"**Model:** {model_choice}")
                                    st.write(f"**Pages referenced:** {len(relevant_chunks) if use_smart_context else max_context_chunks}")
                                    
                            else:
                                st.error(f"❌ {answer}")  # answer contains error message in this case
                                
                                # Suggest fallback
                                st.info("💡 Try using Local Smart Processing mode as a fallback")
                                
                    else:
                        # Enhanced local processing
                        st.markdown("### 🧠 Smart Local Analysis")
                        
                        # Generate intelligent response
                        query_words = query.lower().split()
                        
                        # Find best matching chunks
                        if use_smart_context and relevant_chunks:
                            st.success(f"✅ Found {len(relevant_chunks)} highly relevant sections")
                            
                            answer_parts = []
                            for chunk in relevant_chunks:
                                # Extract relevant sentences
                                sentences = re.split(r'[.!?]+', chunk['text'])
                                relevant_sentences = []
                                
                                for sentence in sentences:
                                    sentence_score = calculate_relevance_score(sentence, query)
                                    if sentence_score > relevance_threshold:
                                        relevant_sentences.append(sentence.strip())
                                
                                if relevant_sentences:
                                    page_answer = f"**From Page {chunk['page']}:**\n" + " ".join(relevant_sentences[:2])
                                    answer_parts.append(page_answer)
                            
                            if answer_parts:
                                st.write("\n\n".join(answer_parts))
                            else:
                                st.write("The document contains information related to your query, but I couldn't extract specific relevant sentences. Try using the AI mode for better analysis.")
                        else:
                            # Fallback to basic keyword matching
                            matched_content = []
                            for chunk in text_chunks[:3]:
                                if any(word in chunk['text'].lower() for word in query_words):
                                    matched_content.append(f"**Page {chunk['page']}:** {chunk['text'][:300]}...")
                            
                            if matched_content:
                                st.write("**Found relevant content:**\n\n" + "\n\n".join(matched_content))
                            else:
                                st.write("No specific matches found. Here's a general overview:")
                                st.write(text_chunks[0]['text'][:500] + "...")
                        
                        st.info("💡 **Tip:** Use the OpenAI API mode for more intelligent and detailed responses!")
    else:
        st.error("❌ No text could be extracted from the PDF. Please try a different file.")
else:
    st.info("👆 Please upload a PDF file to get started")

# Footer with tips
st.markdown("---")
st.markdown("### 💡 Tips for Better Results:")
st.markdown("""
- **🎯 Be specific:** Ask detailed questions rather than general ones
- **🔍 Use keywords:** Include important terms from the document
- **📊 Try different modes:** Compare local vs AI analysis
- **⚙️ Adjust settings:** Fine-tune relevance threshold and context chunks
- **🧠 Smart context:** Enable for better relevance matching
- **📝 Question types:** Try definitions, comparisons, how-to questions, and lists
""")

# Add session state cleanup
if 'query_input' in st.session_state and st.session_state.main_query != st.session_state.get('query_input', ''):
    del st.session_state.query_input
