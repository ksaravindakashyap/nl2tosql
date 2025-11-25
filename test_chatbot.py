import gradio as gr

def test_chatbot():
    """Test the chatbot format fixes."""
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(label="Conversation History")
        msg = gr.Textbox(label="Your Message")
        history = gr.State([])

        def user_message(message, history):
            # Test the dict format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"You said: {message}"})
            return "", history

        msg.submit(user_message, inputs=[msg, history], outputs=[msg, chatbot])

    return demo

if __name__ == '__main__':
    demo = test_chatbot()
    demo.launch(server_name='127.0.0.1', server_port=7860, share=False)