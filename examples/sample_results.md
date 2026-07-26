# Sample Smokebench Report

_Generated: 2026-07-19T14:30:00_

## Results

| Task / Model | gpt-4o-mini | gpt-4o | claude-3-5-sonnet |
|---|---|---|---|
| code_explanation | 90% / 42.1tps / $0.0012 | 95% / 38.7tps / $0.0045 | 92% / 40.2tps / $0.0038 |
| code_generation | 73% / 1% / 35.2tps / $0.0008 | 87% / 32.1tps / $0.0031 | 80% / 36.5tps / $0.0026 |
| creative_writing | 78% / 28.4tps / $0.0011 | 85% / 26.7tps / $0.0042 | 88% / 29.1tps / $0.0035 |
| instruction_json | 95% / 45.3tps / $0.0006 | 98% / 41.2tps / $0.0028 | 97% / 43.8tps / $0.0023 |
| latency | 100% / 142.3tps / $0.0001 | 100% / 138.7tps / $0.0002 | 100% / 140.1tps / $0.0002 |
| long_context | 60% / 31.7tps / $0.0015 | 85% / 29.4tps / $0.0056 | 80% / 30.2tps / $0.0047 |
| math_reasoning | 75% / 38.9tps / $0.0009 | 90% / 35.4tps / $0.0038 | 85% / 37.1tps / $0.0032 |
| summarization | 82% / 41.6tps / $0.0007 | 88% / 38.9tps / $0.0031 | 85% / 40.3tps / $0.0026 |

## Recommendations

| Category | Best model | Aggregate score | Detail |
|---|---|---|---|
| best_overall | gpt-4o | 0.885 | (0.90, 0.87, 0.88, 0.98, 0.85) across math_reasoning,code_generation,summarization,instruction_json,long_context |
| best_coding | gpt-4o | 0.910 | (0.87, 0.95) across code_generation,code_explanation |
| best_reasoning | gpt-4o | 0.940 | (0.90, 0.98) across math_reasoning,instruction_json |
| best_long_context | gpt-4o | 0.850 | (0.85) across long_context |
| best_json_mode | gpt-4o | 0.980 | (0.98) across instruction_json |
| best_writing | claude-3-5-sonnet | 0.900 | (0.88, 0.92) across creative_writing,code_explanation |
| fastest | gpt-4o-mini | 142.300 | (142.3) across latency |
| cheapest | gpt-4o-mini | -0.001 | (-0.0009, -0.0008, -0.0007) across math_reasoning,code_generation,summarization |

## Errors

None.