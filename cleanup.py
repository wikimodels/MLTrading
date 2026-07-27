import os, re
files = ['config_loader.py', 'main.py', 'scheduler.py', 'backtest/engine.py', 'data/collector.py', 'data/processor.py', 'data/storage.py', 'features/engineer.py', 'features/market_breadth.py', 'labeling/triple_barrier.py', 'models/predictor.py', 'models/trainer.py', 'trading/bot.py', 'trading/executor.py', 'trading/monitor.py', 'trading/risk_manager.py']

def remove_russian(text):
    text = re.sub(r'\"\"\"[\s\S]*?[А-Яа-яЁё][\s\S]*?\"\"\"', '\"\"\"[Translated]\"\"\"', text)
    text = re.sub(r'^\s*#.*[А-Яа-яЁё].*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'  #.*[А-Яа-яЁё].*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\"[^\"]*[А-Яа-яЁё][^\"]*\"', '\"\"', text)
    text = re.sub(r'\'[^\']*[А-Яа-яЁё][^\']*\'', '\"\"', text)
    return text

for f in files:
    with open(f, 'r', encoding='utf-8') as file: content = file.read()
    with open(f, 'w', encoding='utf-8') as file: file.write(remove_russian(content))
