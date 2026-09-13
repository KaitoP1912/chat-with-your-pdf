import csv

checks = [
    ('page_aware', 'test_10'),
    ('page_aware', 'test_25'),
    ('fixed_size', 'test_25'),
    ('fixed_size', 'test_20'),
]

for config, qid in checks:
    path = f'results/tuan6_pilot/test_qa_results_{config}.csv'
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['id'] == qid:
                label = r.get('answer_correctness_manual')
                print(f'{config} / {qid}: {label}')