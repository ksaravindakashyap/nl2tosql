"""
Spider Evaluation Module - Phase 3
Simplified evaluation without external dependencies
"""
import json
import sqlite3
from pathlib import Path
import re


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    sql = sql.lower().strip()
    # Remove comments
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    # Normalize whitespace
    sql = ' '.join(sql.split())
    # Remove trailing semicolon
    sql = sql.rstrip(';')
    return sql


def execute_sql(db_path: str, sql: str) -> tuple:
    """Execute SQL and return (success, result_set)."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results
    except Exception as e:
        return False, str(e)


def evaluate_predictions(gold_file: str, pred_file: str, db_dir: str) -> dict:
    """
    Evaluate predictions against gold standard.
    Returns dict with exact_match and execution_accuracy.
    """
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    # Create lookup by question
    pred_dict = {item['question']: item['sql'] for item in pred_data}
    
    exact_matches = 0
    exec_matches = 0
    total = 0
    
    for gold_item in gold_data:
        question = gold_item['question']
        gold_sql = normalize_sql(gold_item['query'])
        db_id = gold_item['db_id']
        
        if question not in pred_dict:
            total += 1
            continue
        
        pred_sql = normalize_sql(pred_dict[question])
        
        # Exact match
        if gold_sql == pred_sql:
            exact_matches += 1
            exec_matches += 1
        else:
            # Execution accuracy
            db_path = Path(db_dir) / db_id / f"{db_id}.sqlite"
            if db_path.exists():
                gold_success, gold_result = execute_sql(str(db_path), gold_item['query'])
                pred_success, pred_result = execute_sql(str(db_path), pred_dict[question])
                
                if gold_success and pred_success and gold_result == pred_result:
                    exec_matches += 1
        
        total += 1
    
    return {
        'exact_match': exact_matches / total if total > 0 else 0,
        'execution_accuracy': exec_matches / total if total > 0 else 0,
        'total': total,
        'exact_count': exact_matches,
        'exec_count': exec_matches
    }


def generate_predictions_from_dev(
    llm, vectorstore, databases, db_descriptions, 
    router_chain, db_list, dev_file: str, output_file: str,
    max_samples: int = 100
):
    """
    Generate predictions for Spider dev set.
    Returns predictions as list of dicts.
    """
    from main import predict_db_id, generate_sql
    
    with open(dev_file, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    predictions = []
    
    for i, item in enumerate(dev_data[:max_samples]):
        question = item['question']
        gold_db_id = item['db_id']
        
        try:
            # Predict database
            pred_db_id = predict_db_id(router_chain, db_list, question)
            if not pred_db_id or pred_db_id not in databases:
                pred_db_id = gold_db_id  # Fallback to gold
            
            # Generate SQL
            sql = generate_sql(llm, vectorstore, databases, db_descriptions, 
                             router_chain, db_list, question, pred_db_id)
            
            predictions.append({
                'question': question,
                'sql': sql,
                'db_id': pred_db_id,
                'gold_db_id': gold_db_id
            })
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{max_samples} predictions...")
        
        except Exception as e:
            print(f"Error on question {i}: {e}")
            predictions.append({
                'question': question,
                'sql': '',
                'db_id': gold_db_id,
                'gold_db_id': gold_db_id,
                'error': str(e)
            })
    
    # Save predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    return predictions
