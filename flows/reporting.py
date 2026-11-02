"""Deterministic human-readable projection of the immutable forecast record."""
def forecast_report(result: dict) -> str:
    lines=['# Research forecast', '', f"Event: {result['event_id']}",
           f"Mode: {result['mode']}; cutoff: {result['as_of_utc']}",
           f"Model: {result['model_id']}; snapshot: {result['snapshot_id']}", '',
           'Ordinal ranking scores are not race times or win odds.', '',
           '| Rank | Driver | Score |', '| ---: | --- | ---: |']
    for row in result['ranking']:
        # Identifiers are data, not instructions or source HTML.
        driver=str(row['driver_id']).replace('|','').replace('\n',' ')
        lines.append(f"| {row['predicted_rank']} | {driver} | {row['prediction_score']:.3f} |")
    lines+=['','## Evidence and limitations','']+[f'- {warning}' for warning in result['warnings']]
    lines+=['', 'Sources: '+', '.join(result['provenance']['source_ids'])]
    return '\n'.join(lines)

