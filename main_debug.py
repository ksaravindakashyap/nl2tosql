import os
import json
import sqlite3
import pandas as pd
import gradio as gr

SPIDER_PATH = './spider'
TRAIN_FILE = os.path.join(SPIDER_PATH, 'train_spider.json')
DB_FOLDER = os.path.join(SPIDER_PATH, 'database')

with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
    spider_data = json.load(f)

# simple DB list
db_ids = sorted({d['db_id'] for d in spider_data})

# simple placeholder generate/execute

def get_db_id(question):
    if not question:
        return None
    lower = question.lower()
    if 'in database' in lower:
        return lower.split('in database')[1].split(',')[0].strip()
    return None


def generate_sql(question):
    db = get_db_id(question)
    if not db:
        return "-- specify 'In database <db_id>, ...'"
    return "SELECT name FROM sqlite_master WHERE type='table' LIMIT 5;"


def execute_sql(sql, db_id):
    try:
        conn = sqlite3.connect(os.path.join(DB_FOLDER, db_id, f"{db_id}.db"))
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df.to_string()
    except Exception as e:
        return str(e)

with gr.Blocks(title='Debug Gradio') as demo:
    q = gr.Textbox(label='Question')
    gen = gr.Button('Generate SQL')
    sql_out = gr.Textbox(label='SQL', interactive=True)
    exec_btn = gr.Button('Execute')
    res = gr.Textbox(label='Results')

    gen.click(lambda question: (generate_sql(question), get_db_id(question)), inputs=q, outputs=[sql_out, gr.State('')])
    exec_btn.click(lambda sql, db: execute_sql(sql, db), inputs=[sql_out, gr.State('')], outputs=res)

if __name__ == '__main__':
    print('Starting debug Gradio on 127.0.0.1:7860')
    demo.launch(server_name='127.0.0.1', server_port=7860, share=False)
