import json
import requests

import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential

# Page config
st.set_page_config(page_title="Financial News Chatbot", page_icon="💰", layout="wide")

st.title("💰 Financial News Chatbot")
st.markdown("Hỏi đáp thông tin tài chính Việt Nam với AI")

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def send_chat_request(user_message, conversation_history=None):
    """Send chat request to backend API"""
    url = f"{API_BASE_URL}/chat"

    payload = json.dumps(
        {"message": user_message, "conversation_history": conversation_history or []}
    )

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=30)
        if response.status_code != 200:
            raise Exception(
                f"API request failed: {response.status_code} - {response.text}"
            )
        return response.json()
    except requests.exceptions.Timeout:
        raise Exception("Request timeout. Server is taking too long to respond.")
    except requests.exceptions.ConnectionError:
        raise Exception("Cannot connect to server. Please check if backend is running.")


def check_api_health():
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


# API Status
api_online = check_api_health()
if api_online:
    st.success("🟢 API Online - Ready to chat!")
else:
    st.error("🔴 API Offline - Please start the backend server")
    st.stop()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


# Sidebar with info and controls
with st.sidebar:
    st.markdown("### 📊 Chat Statistics")
    st.markdown(f"**Messages:** {len(st.session_state.messages)}")

    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            # References
            references = metadata.get("references", [])
            if references:
                st.markdown("**📚 References:**")
                for i, ref in enumerate(references[:3], 1):  # Show max 3 references
                    with st.expander(
                        f"Reference {i}: {ref.get('title', 'N/A')[:50]}..."
                    ):
                        st.markdown(f"**Source:** {ref.get('source', 'N/A')}")
                        st.markdown(f"**URL:** {ref.get('url', 'N/A')}")
                        if ref.get("relevance_score"):
                            st.markdown(
                                f"**Relevance:** {ref.get('relevance_score', 0.0):.2f}"
                            )


# Handle sample question selection
if hasattr(st.session_state, "sample_question"):
    prompt = st.session_state.sample_question
    del st.session_state.sample_question
else:
    prompt = None


# Accept user input
if not prompt:
    prompt = st.chat_input("Type your question...")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Add to conversation history for API
    st.session_state.conversation_history.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        try:
            with st.spinner("🤔 Thinking..."):
                # Get response from API
                response_data = send_chat_request(
                    prompt, st.session_state.conversation_history[-10:]
                )

                answer = response_data.get(
                    "answer", "Xin lỗi, tôi không thể trả lời câu hỏi này."
                )
                references = response_data.get("references", [])
                # Display answer
                st.markdown(answer)
                # Display references
                if references:
                    st.markdown("**📚 Tài liệu tham khảo:**")
                    for i, ref in enumerate(references[:3], 1):
                        with st.expander(
                            f"Tài liệu {i}: {ref.get('title', 'N/A')[:50]}..."
                        ):
                            st.markdown(f"**Nguồn:** {ref.get('source', 'N/A')}")
                            if ref.get("url"):
                                st.markdown(
                                    f"**URL:** [Xem chi tiết]({ref.get('url')})"
                                )

                # Add assistant response to chat history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "metadata": {
                            "references": references,
                        },
                    }
                )

                # Add to conversation history
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": answer}
                )

        except Exception as e:
            error_message = f"❌ Lỗi: {str(e)}"
            st.error(error_message)

            # Add error message to chat history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "metadata": {"references": []},
                }
            )


# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666;'>
    💰 Financial News Chatbot - Powered by AI & RAG<br>
    📰 Tin tức tài chính Việt Nam | 🔍 Tìm kiếm thông minh
</div>
""",
    unsafe_allow_html=True,
)
