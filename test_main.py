import os
import json
import gradio as gr
import re

# Mock data for testing
DB_DESCRIPTIONS = {
    "concert_singer": "Concerts, singers, stadiums, and performances",
    "pets_1": "Pet owners, pets (dogs, cats, rabbits), breeds and treatments",
}

databases = {"concert_singer": "mock_db", "pets_1": "mock_db"}

def mock_llm():
    """Mock LLM for testing."""
    class MockLLM:
        def invoke(self, prompt):
            return "SELECT * FROM table LIMIT 5"
    return MockLLM()

def mock_vectorstore():
    """Mock vectorstore for testing."""
    class MockVectorstore:
        def similarity_search(self, question, k=5, filter=None):
            return []
    return MockVectorstore()

def mock_router_chain():
    """Mock router chain for testing."""
    class MockChain:
        def invoke(self, data):
            return type('Result', (), {'db_id': 'concert_singer'})()
    return MockChain(), "mock_db_list"

def build_gradio_app(llm, vectorstore, databases, db_descriptions, router_chain, db_list):
    """Construct and return the Gradio Blocks app for testing chatbot format."""
    with gr.Blocks(title="NL2SQL Test") as demo:
        gr.Markdown("# NL2SQL System - Test")
        gr.Markdown("Testing chatbot format fixes.")

        chatbot = gr.Chatbot(label="Conversation History")
        msg = gr.Textbox(label="Your Message", placeholder="Ask a question or type /schema")
        sql_box = gr.Textbox(label="Generated SQL (Edit if needed)", lines=6, interactive=True, visible=False)
        with gr.Row():
            regenerate_btn = gr.Button("Regenerate SQL", visible=False)
            run_query_btn = gr.Button("Run Query", visible=False)

        # State for history, current db_id, etc.
        history = gr.State([])
        current_db_id = gr.State("")

        def user_message(message, history, current_db_id):
            """Handle user input: /schema, force DB switch, or generate SQL with auto DB detection."""
            message = message.strip()
            if message == "/schema":
                if not current_db_id:
                    response = "No database selected. Ask a question first."
                else:
                    response = f"Current Database: {current_db_id}\nMock schema info."
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                return "", history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), current_db_id
            elif message.lower().startswith("use database"):
                parts = message.lower().split("use database")
                new_db_id = parts[1].strip()
                if new_db_id in databases:
                    current_db_id = new_db_id
                    response = f"Switched to database: {new_db_id}"
                else:
                    response = f"Database {new_db_id} not found."
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                return "", history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), current_db_id
            else:
                # Mock SQL generation
                sql = "SELECT * FROM mock_table WHERE id = 1;"
                response = f"Generated SQL:\n{sql}"
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                return "", history, gr.update(value=sql, visible=True), gr.update(visible=True), gr.update(visible=True), current_db_id

        def regenerate_sql(history, db_id, question):
            """Regenerate SQL for the current question."""
            sql = "SELECT * FROM mock_table WHERE id = 2;"
            # Update history
            history[-1] = {"role": "assistant", "content": f"Regenerated SQL:\n{sql}"}
            return history, gr.update(value=sql, visible=True), gr.update(visible=True), gr.update(visible=True)

        def run_query(sql, db_id, history):
            """Execute the SQL and append to history."""
            response = f"Mock execution result for: {sql}"
            history.append({"role": "user", "content": "Executed SQL"})
            history.append({"role": "assistant", "content": response})
            return history, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        msg.submit(user_message, inputs=[msg, history, current_db_id], outputs=[msg, chatbot, sql_box, regenerate_btn, run_query_btn, current_db_id])
        regenerate_btn.click(regenerate_sql, inputs=[history, current_db_id, gr.State("")], outputs=[chatbot, sql_box, regenerate_btn, run_query_btn])
        run_query_btn.click(run_query, inputs=[sql_box, current_db_id, history], outputs=[chatbot, sql_box, regenerate_btn, run_query_btn])

    return demo

if __name__ == '__main__':
    print('Testing NL2SQL chatbot format...')
    llm = mock_llm()
    vectorstore = mock_vectorstore()
    router_chain, db_list = mock_router_chain()
    demo = build_gradio_app(llm, vectorstore, databases, DB_DESCRIPTIONS, router_chain, db_list)

    demo.launch(server_name='127.0.0.1', server_port=7861, share=False)