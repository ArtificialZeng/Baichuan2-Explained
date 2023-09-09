# 导入streamlit库
import streamlit as st  
# 从transformers库中导入用于因果语言模型的自动模型和自动分词器
from transformers import AutoModelForCausalLM, AutoTokenizer  
# 从transformers库中导入生成配置工具
from transformers.generation.utils import GenerationConfig  

# 设置Streamlit页面的标题
st.set_page_config(page_title="Baichuan 2")  
st.title("Baichuan 2")  

# 使用@st.cache_resource装饰器缓存该函数的结果
@st.cache_resource  
def init_model():
    # 从预训练模型"baichuan-inc/Baichuan2-13B-Chat"加载模型
    model = AutoModelForCausalLM.from_pretrained(
        "baichuan-inc/Baichuan2-13B-Chat",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        "baichuan-inc/Baichuan2-13B-Chat"
    )
    # 从预训练模型"baichuan-inc/Baichuan2-13B-Chat"加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        "baichuan-inc/Baichuan2-13B-Chat",
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer

# 定义了一个清空会话历史的方法
def clear_chat_history():
    del st.session_state.messages

# 初始化或展示聊天历史
def init_chat_history():
    # 发送问候消息
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是百川大模型，很高兴为您服务🥰")
    # 展示聊天记录
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages

# 定义应用的主要逻辑
def main():
    # 初始化模型和分词器
    model, tokenizer = init_model()
    # 初始化或获取聊天历史
    messages = init_chat_history()

    # 处理用户输入
    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            for response in model.chat(tokenizer, messages, stream=True):
                placeholder.markdown(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

        # 提供一个按钮供用户点击，以清空对话历史
        st.button("清空对话", on_click=clear_chat_history)

# 当脚本作为主程序运行时，执行main函数
if __name__ == "__main__":
    main()
